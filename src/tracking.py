"""Automatic detection and tracking of every sunspot in a DS0N box (line B).

`03B` separates the spots of a region by hand: one copy-pasted block per spot, a
`CLUSTER_XLIM` typed in pixels, and `'largest'` re-picked independently every frame. This
module replaces that with one pass over the continuum cube that finds the spots itself,
follows them through splits and merges, and hands back umbra / penumbra masks per spot in
the same `data` dict shape `load_ds0n_region` returns — so `analysis`, `spectra` and
`animation` run on each spot unchanged.

The rules, in the order the tracker applies them each frame:

* **A spot** is a connected component (8-connectivity) of the footprint ``cont <
  penumbra_thresh`` that contains umbra (``cont < umbra_thresh``). Footprint components
  under `MIN_CLUSTER_PX` are granulation and are dropped before anything else.
* **Initial detection.** In the first frame with data, up to `max_initial_spots` footprint
  components of at least `min_spot_px` that contain umbra, largest first: ``A``, ``B``, …
  A component that already holds several umbrae of at least `min_umbra_split_px` gets one
  spot per umbra straight away, named as a split (``A``, ``AA``, ``AB``).
* **Following** is by pixel overlap with the spot's own umbra in its last frame, or with
  its footprint when it has no umbra. A spot missing for more than `max_missing_frames`
  frames with data ends as ``lost``. All-NaN frames are skipped and break nothing.
* **Umbra split.** When a spot's umbra overlaps two or more umbra components, the one it
  overlaps most keeps the name and the largest other one, if it measures at least
  `min_umbra_split_px` and `min_split_fraction` of the parent's umbra, and stays apart for
  `min_split_frames` consecutive frames, becomes a
  child named ``<parent><letter>`` (``A`` -> ``AA``, ``AB``; ``AA`` -> ``AAA``), dated back
  to the first of those frames. Without the persistence test the umbra boundary flickering
  across the threshold would spawn a child every few frames.
* **Shared penumbra.** Every spot's penumbra is the non-umbra part of the footprint
  component(s) its umbra lies in. A parent and child in the same component therefore share
  the same penumbra pixels; once their umbrae sit in different components for
  `min_split_frames` frames, the child is marked ``penumbra_split``.
* **Merge.** Two spots claiming the same umbra component — or a spot without umbra sitting
  in another spot's footprint component — for `min_split_frames` consecutive frames merge.
  An ancestor always survives its own descendant (``A`` over ``AA`` over ``AAB``);
  otherwise the larger survives. The loser ends at the frame the contact began, and the
  umbra pixels it held during the contact frames go to the survivor. While a contact is
  still pending the contested umbra is split between the spots by nearest previous umbra.
* **Filters**, after the whole cube: a child whose penumbra split must have a median area
  of at least `min_spot_px` and last at least `min_duration_h`. Children that only ever
  split their umbra are exempt, and are shown with a trailing ``'`` (``DS00_AA'``; ``p`` on
  disk, ``DS00_AAp``) — but one that rejoins within `heal_within_h` is annulled outright
  ("healed"): it was a light bridge or a gap between umbral dots, not a spot. The initial spots are exempt too, having passed the size test to be
  detected at all.

A spot that *emerges* mid-window without splitting off a tracked one is not followed. The
tracker counts how many frames held such an untracked spot-sized component, so the size of
that blind spot is reported rather than silent.
"""

from __future__ import annotations

import dataclasses
import pathlib
import string
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import ndimage

from .config import DS0N_DEFAULTS, MIN_CLUSTER_PX
from .plotting import save_figure


@dataclasses.dataclass
class TrackingParams:
    """Every threshold the tracker uses. Defaults are the ones agreed for the test run."""

    umbra_thresh: float = DS0N_DEFAULTS['umbra_thresh']
    penumbra_thresh: float = DS0N_DEFAULTS['penumbra_thresh']
    max_initial_spots: int = 4
    # ~4 MSH at HMI's 0.5 arcsec/px: generous enough for small spots, above a lone pore.
    min_spot_px: int = 100
    # Defaults after a sweep over DS00-DS09: 10 px / 3 frames / no fraction gave 108 kept
    # spots (63 of them ' splits of umbral dots and light bridges); these give 30.
    min_umbra_split_px: int = 80
    # ... and at least this fraction of the parent's whole umbra in that frame, so an umbral
    # dot beside a large umbra is not a split. 0 disables it.
    min_split_fraction: float = 0.25
    # Consecutive frames a split, a penumbra split or a merge must persist (20 x 720 s = 4 h).
    min_split_frames: int = 20
    min_duration_h: float = 12.0
    # An umbra-only split that rejoins within this many hours is annulled (see `_heal`).
    # None = `min_duration_h`.
    heal_within_h: float | None = 24.0
    min_cluster_px: int = MIN_CLUSTER_PX
    max_missing_frames: int = 3
    connectivity: int = 2


@dataclasses.dataclass
class Spot:
    name: str                      # suffix only: 'A', 'AA', ...
    parent: str | None
    t_start: int
    frames: dict = dataclasses.field(default_factory=dict)   # t -> (umbra_idx, penumbra_idx)
    active: bool = True
    end_reason: str = ''
    penumbra_split: bool = False
    missing: int = 0
    n_children: int = 0
    split_cand: dict | None = None
    pen_sep_count: int = 0
    # Filled by `_finalise`.
    kept: bool = True
    reject_reason: str = ''

    @property
    def t_end(self) -> int:
        return max(self.frames)

    @property
    def prime(self) -> bool:
        """A child that only ever split its umbra, never its penumbra."""
        return self.parent is not None and not self.penumbra_split

    def last(self) -> tuple[np.ndarray, np.ndarray]:
        return self.frames[self.t_end]


def display_name(ds_name: str, spot: Spot) -> str:
    return f"{ds_name}_{spot.name}{chr(39) if spot.prime else ''}"


def dir_name(ds_name: str, spot: Spot) -> str:
    return f"{ds_name}_{spot.name}{'p' if spot.prime else ''}"


# ── per-frame segmentation ──────────────────────────────────────────────────

@dataclasses.dataclass
class _Frame:
    foot_lab: np.ndarray
    umb_lab: np.ndarray
    n_umb: int
    umb_size: np.ndarray       # (n_umb + 1,) pixel count per umbra label
    umb_foot: np.ndarray       # (n_umb + 1,) footprint label each umbra label lies in
    foot_size: np.ndarray
    umb_obj: list
    foot_obj: list


def _segment(frame: np.ndarray, p: TrackingParams, structure) -> _Frame:
    finite = np.isfinite(frame)
    with np.errstate(invalid='ignore'):
        foot = (frame < p.penumbra_thresh) & finite
    foot_lab, n_foot = ndimage.label(foot, structure=structure)
    foot_size = np.bincount(foot_lab.ravel(), minlength=n_foot + 1)
    small = foot_size < p.min_cluster_px
    small[0] = False
    if small.any():
        foot_lab[small[foot_lab]] = 0
        foot_size[small] = 0

    with np.errstate(invalid='ignore'):
        umb = (frame < p.umbra_thresh) & (foot_lab > 0)
    umb_lab, n_umb = ndimage.label(umb, structure=structure)
    umb_size = np.bincount(umb_lab.ravel(), minlength=n_umb + 1)
    umb_foot = np.zeros(n_umb + 1, dtype=int)
    if n_umb:
        # Same connectivity and umbra is a subset of the footprint, so each umbra component
        # lies inside exactly one footprint component.
        umb_foot[1:] = ndimage.maximum(foot_lab, umb_lab, np.arange(1, n_umb + 1))
    return _Frame(foot_lab, umb_lab, n_umb, umb_size, umb_foot, foot_size,
                  ndimage.find_objects(umb_lab), ndimage.find_objects(foot_lab))


def _local_idx(local_mask: np.ndarray, sl, nx: int) -> np.ndarray:
    yy, xx = np.nonzero(local_mask)
    return ((yy + sl[0].start) * nx + xx + sl[1].start).astype(np.int32)


def _umb_idx(fr: _Frame, comp: int, nx: int) -> np.ndarray:
    sl = fr.umb_obj[comp - 1]
    return _local_idx(fr.umb_lab[sl] == comp, sl, nx)


def _pen_idx(fr: _Frame, foot_comp: int, nx: int) -> np.ndarray:
    sl = fr.foot_obj[foot_comp - 1]
    return _local_idx((fr.foot_lab[sl] == foot_comp) & (fr.umb_lab[sl] == 0), sl, nx)


def _centroid(idx: np.ndarray, nx: int) -> np.ndarray:
    return np.array([np.mean(idx // nx), np.mean(idx % nx)])


# ── the tracker ─────────────────────────────────────────────────────────────

class _Tracker:
    def __init__(self, shape, time_h, params: TrackingParams):
        self.p = params
        self.time_h = np.asarray(time_h)
        self.ny, self.nx = shape
        self.structure = ndimage.generate_binary_structure(2, params.connectivity)
        self.spots: dict[str, Spot] = {}
        self.events: list[dict] = []
        self.merge_pending: dict[frozenset, list] = {}     # pair -> [count, first_frame]
        self.untracked = []                                 # per frame with data
        self.started = False

    # -- naming and genealogy ------------------------------------------------

    def _child_name(self, parent: Spot) -> str:
        """Provisional: healed splits consume names too, so `_rename` relabels at the end."""
        name = parent.name + _letter(parent.n_children)
        parent.n_children += 1
        return name

    def _is_ancestor(self, a: str, b: str) -> bool:
        """Whether spot `a` is an ancestor of spot `b`."""
        parent = self.spots[b].parent
        while parent is not None:
            if parent == a:
                return True
            parent = self.spots[parent].parent
        return False

    def _winner(self, a: str, b: str) -> tuple[str, str]:
        if self._is_ancestor(a, b):
            return a, b
        if self._is_ancestor(b, a):
            return b, a
        size = {n: sum(len(i) for i in self.spots[n].last()) for n in (a, b)}
        return (a, b) if size[a] >= size[b] else (b, a)

    def _event(self, t, kind, spot, other='', detail=''):
        self.events.append(dict(frame=t, event=kind, spot=spot, other=other, detail=detail))

    # -- initial detection ---------------------------------------------------

    def _detect(self, t: int, fr: _Frame):
        p = self.p
        with_umbra = set(fr.umb_foot[1:].tolist())
        candidates = [c for c in range(1, len(fr.foot_size))
                      if fr.foot_size[c] >= p.min_spot_px and c in with_umbra]
        candidates.sort(key=lambda c: -fr.foot_size[c])
        for letter, comp in zip(string.ascii_uppercase, candidates[:p.max_initial_spots]):
            umbrae = sorted((u for u in range(1, fr.n_umb + 1) if fr.umb_foot[u] == comp),
                            key=lambda u: -fr.umb_size[u])
            total = sum(fr.umb_size[u] for u in umbrae)
            big = [u for u in umbrae[1:] if fr.umb_size[u] >= p.min_umbra_split_px
                   and fr.umb_size[u] >= p.min_split_fraction * total]
            main = [u for u in umbrae if u not in big]
            pen = _pen_idx(fr, comp, self.nx)
            spot = Spot(letter, None, t)
            spot.frames[t] = (np.concatenate([_umb_idx(fr, u, self.nx) for u in main]), pen)
            self.spots[letter] = spot
            self._event(t, 'detected', letter, detail=f'{fr.foot_size[comp]} px')
            for u in big:
                child = Spot(self._child_name(spot), letter, t)
                child.frames[t] = (_umb_idx(fr, u, self.nx), pen)
                self.spots[child.name] = child
                self._event(t, 'split_umbra', child.name, letter, 'present at detection')
        self.started = bool(self.spots)

    # -- one frame -----------------------------------------------------------

    def step(self, t: int, frame: np.ndarray):
        if not np.isfinite(frame).any():
            return
        fr = _segment(frame, self.p, self.structure)
        if not self.started:
            self._detect(t, fr)
            return

        p, nx = self.p, self.nx
        active = [s for s in self.spots.values() if s.active]

        # 1. Overlap of every spot's last masks with this frame's components.
        claims: dict[str, dict[int, int]] = {}      # spot -> {umbra comp: overlap}
        foot_target: dict[str, int] = {}             # umbra-less spot -> footprint comp
        for s in active:
            umb_prev, pen_prev = s.last()
            ov_u = np.bincount(fr.umb_lab.ravel()[umb_prev], minlength=fr.n_umb + 1)
            ov_u[0] = 0
            if ov_u.any():
                claims[s.name] = {int(c): int(ov_u[c]) for c in np.flatnonzero(ov_u)}
                continue
            foot_prev = np.concatenate([umb_prev, pen_prev])
            ov_f = np.bincount(fr.foot_lab.ravel()[foot_prev], minlength=len(fr.foot_size))
            ov_f[0] = 0
            if ov_f.any():
                foot_target[s.name] = int(ov_f.argmax())
            else:
                s.missing += 1
                if s.missing > p.max_missing_frames:
                    s.active = False
                    s.end_reason = 'lost'
                    self._event(t, 'lost', s.name)

        # 2. Unclaimed umbra components inside a tracked footprint component join the spot
        #    there (nearest one if several) — an umbra re-forming, or a new core appearing.
        claimed = {c for cl in claims.values() for c in cl}
        members = self._members(fr, claims, foot_target)
        for u in range(1, fr.n_umb + 1):
            if u in claimed:
                continue
            here = members.get(int(fr.umb_foot[u]), [])
            if not here:
                continue
            if len(here) == 1:
                owner = here[0]
            else:
                c = _centroid(_umb_idx(fr, u, nx), nx)
                owner = min(here, key=lambda n: np.hypot(
                    *(c - _centroid(np.concatenate(self.spots[n].last()), nx))))
            claims.setdefault(owner, {})[u] = 0
            foot_target.pop(owner, None)
        members = self._members(fr, claims, foot_target)

        # 3. Contacts: the same umbra component claimed twice, or an umbra-less spot in a
        #    footprint component someone else is in. Persisting ones become merges.
        claimants: dict[int, list[str]] = {}
        for name, cl in claims.items():
            for c in cl:
                claimants.setdefault(c, []).append(name)
        pairs = set()
        for names in claimants.values():
            pairs |= {frozenset((a, b)) for i, a in enumerate(names) for b in names[i + 1:]}
        for name, comp in foot_target.items():
            pairs |= {frozenset((name, o)) for o in members.get(comp, []) if o != name}
        self.merge_pending = {pair: ([self.merge_pending[pair][0] + 1,
                                      self.merge_pending[pair][1]]
                                     if pair in self.merge_pending else [1, t])
                              for pair in pairs}
        def order(item):
            # Earliest contact first; on a tie a direct parent-child pair before any other,
            # so a child touching both its parent and a grandparent rejoins its parent. The
            # names close the tie so the result never depends on set iteration order.
            pair, (_, first) = item
            a, b = sorted(pair)
            direct = self.spots[a].parent == b or self.spots[b].parent == a
            return first, not direct, a, b

        for pair, (count, first) in sorted(self.merge_pending.items(), key=order):
            a, b = sorted(pair)
            if count < p.min_split_frames or not (self.spots[a].active and self.spots[b].active):
                continue
            win, lose = self._winner(a, b)
            self._merge(win, lose, first, t)
            for c in claims.pop(lose, {}):
                claims.setdefault(win, {}).setdefault(c, 0)
            foot_target.pop(lose, None)
            for names in claimants.values():
                if lose in names:
                    names[names.index(lose)] = win
                    names[:] = list(dict.fromkeys(names))
        self.merge_pending = {pair: v for pair, v in self.merge_pending.items()
                              if all(self.spots[n].active for n in pair)}

        # 4. Contested umbra still pending: split it by nearest previous umbra.
        shares: dict[str, list[np.ndarray]] = {}
        contested = {c: names for c, names in claimants.items() if len(names) > 1}
        for c, names in contested.items():
            for name, idx in self._partition(fr, c, names).items():
                shares.setdefault(name, []).append(idx)

        # 5. Umbra splits, with the persistence test.
        new_children: dict[str, list[int]] = {}
        for name in list(claims):
            s = self.spots[name]
            if not s.active:
                continue
            own = sorted((c for c in claims[name] if c not in contested),
                         key=lambda c: (-claims[name][c], -fr.umb_size[c]))
            if len(own) < 2:
                s.split_cand = None
                continue
            total = sum(fr.umb_size[c] for c in own)
            rest = [c for c in own[1:] if fr.umb_size[c] >= p.min_umbra_split_px
                    and fr.umb_size[c] >= p.min_split_fraction * total]
            if not rest:
                s.split_cand = None
                continue
            sec = max(rest, key=lambda c: fr.umb_size[c])
            frag = _umb_idx(fr, sec, nx)
            cand = s.split_cand
            if cand is not None and np.intersect1d(frag, cand['history'][-1][1]).size:
                cand['history'].append((t, frag))
            else:
                cand = s.split_cand = dict(history=[(t, frag)])
            cand['comp'] = sec
            if len(cand['history']) >= p.min_split_frames:
                child = Spot(self._child_name(s), name, cand['history'][0][0])
                for tt, idx in cand['history'][:-1]:
                    if tt in s.frames:
                        u_par, pen = s.frames[tt]
                        s.frames[tt] = (np.setdiff1d(u_par, idx, assume_unique=True), pen)
                        child.frames[tt] = (idx, pen)
                self.spots[child.name] = child
                self._event(child.t_start, 'split_umbra', child.name, name,
                            f'{fr.umb_size[sec]} px, confirmed at frame {t}')
                new_children[child.name] = [sec]
                del claims[name][sec]
                s.split_cand = None

        # 5b. Umbra fragments that drifted into a *different* footprint component without
        #     qualifying as a split (too small, or not a pending candidate) are let go, like
        #     a penumbra fragment without umbra. Otherwise the spot keeps a second, unrelated
        #     footprint — on DS02 that inflated spot D from a median 1.9k px to 14k px.
        for name in list(claims):
            s = self.spots[name]
            own = sorted((c for c in claims[name] if c not in contested),
                         key=lambda c: (-claims[name][c], -fr.umb_size[c]))
            if len(own) < 2:
                continue
            home = fr.umb_foot[own[0]]
            pending = s.split_cand['comp'] if s.split_cand else None
            for c in own[1:]:
                if fr.umb_foot[c] != home and c != pending:
                    del claims[name][c]

        # 6. Write this frame's masks.
        foot_of: dict[str, set] = {}
        for name in list(claims) + list(new_children) + list(foot_target):
            s = self.spots[name]
            if not s.active:
                continue
            if name in foot_target:
                comps, umb = {foot_target[name]}, np.array([], dtype=np.int32)
            else:
                own = new_children.get(name) or [c for c in claims[name] if c not in contested]
                parts = [_umb_idx(fr, c, nx) for c in own] + shares.get(name, [])
                umb = np.concatenate(parts) if parts else np.array([], dtype=np.int32)
                comps = {int(fr.umb_foot[c]) for c in own}
                comps |= {int(fr.umb_foot[c]) for c in contested if name in contested[c]}
            if not comps:
                continue
            pen =np.concatenate([_pen_idx(fr, f, nx) for f in sorted(comps)])
            s.frames[t] = (umb, pen)
            s.missing = 0
            foot_of[name] = comps

        # 7. Penumbra separation of children from their (still active) parent.
        for name, comps in foot_of.items():
            s = self.spots[name]
            if s.parent is None or s.penumbra_split or s.parent not in foot_of:
                continue
            if comps.isdisjoint(foot_of[s.parent]):
                s.pen_sep_count += 1
                if s.pen_sep_count >= p.min_split_frames:
                    s.penumbra_split = True
                    self._event(t, 'split_penumbra', name, s.parent)
            else:
                s.pen_sep_count = 0

        tracked = set().union(*foot_of.values()) if foot_of else set()
        with_umbra = set(fr.umb_foot[1:].tolist())
        self.untracked.append(sum(1 for c in range(1, len(fr.foot_size))
                                  if fr.foot_size[c] >= p.min_spot_px and c in with_umbra
                                  and c not in tracked))

    def _members(self, fr, claims, foot_target) -> dict[int, list[str]]:
        members: dict[int, list[str]] = {}
        for name, cl in claims.items():
            for c in cl:
                f = int(fr.umb_foot[c])
                if name not in members.setdefault(f, []):
                    members[f].append(name)
        for name, f in foot_target.items():
            if name not in members.setdefault(f, []):
                members[f].append(name)
        return members

    def _partition(self, fr: _Frame, comp: int, names: list[str], margin: int = 30):
        sl = fr.umb_obj[comp - 1]
        y0, y1 = max(sl[0].start - margin, 0), min(sl[0].stop + margin, self.ny)
        x0, x1 = max(sl[1].start - margin, 0), min(sl[1].stop + margin, self.nx)
        window = (slice(y0, y1), slice(x0, x1))
        local = fr.umb_lab[window] == comp
        dist = []
        for name in names:
            umb_prev, pen_prev = self.spots[name].last()
            prev = umb_prev if umb_prev.size else pen_prev
            seed = np.zeros(local.shape, dtype=bool)
            yy, xx = prev // self.nx - y0, prev % self.nx - x0
            ok = (yy >= 0) & (yy < seed.shape[0]) & (xx >= 0) & (xx < seed.shape[1])
            seed[yy[ok], xx[ok]] = True
            dist.append(ndimage.distance_transform_edt(~seed) if seed.any()
                        else np.full(local.shape, np.inf))
        nearest = np.argmin(np.stack(dist), axis=0)
        return {name: _local_idx(local & (nearest == i), window, self.nx)
                for i, name in enumerate(names)}

    def _merge(self, win: str, lose: str, first: int, t: int):
        w, l = self.spots[win], self.spots[lose]
        heal_h = self.p.min_duration_h if self.p.heal_within_h is None else self.p.heal_within_h
        if l.prime and self.time_h[t] - self.time_h[l.t_start] < heal_h:
            self._heal(w, l, t)
            return
        for tt in [tt for tt in l.frames if tt >= first]:
            u_l, _ = l.frames.pop(tt)
            if tt in w.frames:
                u_w, pen_w = w.frames[tt]
                w.frames[tt] = (np.union1d(u_w, u_l), np.setdiff1d(pen_w, u_l))
        l.active = False
        l.end_reason = f'merged_into:{win}'
        self._event(first, 'merge', lose, win, f'confirmed at frame {t}')

    def _heal(self, w: Spot, l: Spot, t: int):
        """An umbra-only split that closes up again within `min_duration_h` was a light
        bridge or a gap between umbral dots, not a new spot: annul it. Every umbra pixel it
        ever held goes to the spot it rejoined, and its children are re-parented there.

        Without this, real umbrae spawn dozens of short-lived ' spots per region — DS00 gave
        89 confirmed umbra splits in 180 h, nearly all rejoining within hours — and each
        one also cuts a temporary hole into its parent's umbra series.
        """
        for tt, (u_l, pen_l) in l.frames.items():
            if tt in w.frames:
                u_w, pen_w = w.frames[tt]
                w.frames[tt] = (np.union1d(u_w, u_l), np.setdiff1d(pen_w, u_l))
            else:
                w.frames[tt] = (u_l, pen_l)
        for child in self.spots.values():
            if child.parent == l.name:
                child.parent = w.name
        l.frames = {}
        l.active = False
        l.end_reason = f'healed_into:{w.name}'
        self._event(t, 'healed', l.name, w.name,
                    f'umbra split annulled after {self.time_h[t] - self.time_h[l.t_start]:.1f} h')


def _letter(k: int) -> str:
    return string.ascii_uppercase[k] if k < 26 else f'[{k}]'


def _rename(spots: dict[str, Spot]) -> dict[str, str]:
    """Final names: children lettered in birth order among the spots that survived, so the
    healed splits leave no gaps (``AA``, ``AB`` rather than ``AL``, ``AT``). Returns the
    provisional -> final map. A child whose parent left no frames hangs off the nearest
    surviving ancestor."""
    alive = {n: s for n, s in spots.items() if s.frames}
    parent_of = {}
    for name, s in alive.items():
        parent = s.parent
        while parent is not None and parent not in alive:
            parent = spots[parent].parent
        parent_of[name] = parent

    mapping, counters = {}, {}
    pending = sorted(alive, key=lambda n: (alive[n].t_start, len(n), n))
    while pending:
        # Birth order, except that a parent is always named before its children (a heal
        # can re-parent a child onto a spot born after it).
        name = next(n for n in pending if parent_of[n] is None or parent_of[n] in mapping)
        pending.remove(name)
        parent = parent_of[name]
        if parent is None:
            mapping[name] = name
        else:
            k = counters.get(parent, 0)
            counters[parent] = k + 1
            mapping[name] = mapping[parent] + _letter(k)
    for name, s in alive.items():
        s.parent = mapping[parent_of[name]] if parent_of[name] is not None else None
    for name, s in alive.items():
        s.name = mapping[name]
    return mapping


# ── public interface ────────────────────────────────────────────────────────

@dataclasses.dataclass
class TrackResult:
    ds_name: str
    shape: tuple[int, int, int]
    spots: dict[str, Spot]
    events: pd.DataFrame
    table: pd.DataFrame
    untracked: np.ndarray
    params: TrackingParams

    def kept(self) -> list[Spot]:
        return [s for s in self.spots.values() if s.kept]

    def bbox(self, name: str, margin: int = 20) -> tuple[slice, slice]:
        """The spot's bounding box over its whole life, plus `margin` px, clipped to the box."""
        _, ny, nx = self.shape
        idx = np.concatenate([np.concatenate(v) for v in self.spots[name].frames.values()])
        yy, xx = idx // nx, idx % nx
        return (slice(max(int(yy.min()) - margin, 0), min(int(yy.max()) + margin + 1, ny)),
                slice(max(int(xx.min()) - margin, 0), min(int(xx.max()) + margin + 1, nx)))

    def masks(self, name: str, bbox: tuple[slice, slice] | None = None):
        """(umbra, penumbra, present) bool cubes over the spot's window, t_start..t_end,
        cropped to `bbox` (the whole box when None)."""
        s = self.spots[name]
        n_w = s.t_end - s.t_start + 1
        _, ny, nx = self.shape
        ys, xs = bbox or (slice(0, ny), slice(0, nx))
        h, w = ys.stop - ys.start, xs.stop - xs.start

        def local(idx):
            return (idx // nx - ys.start) * w + (idx % nx - xs.start)

        umbra = np.zeros((n_w, h * w), dtype=bool)
        penumbra = np.zeros((n_w, h * w), dtype=bool)
        present = np.zeros(n_w, dtype=bool)
        for t, (u, pen) in s.frames.items():
            umbra[t - s.t_start, local(u)] = True
            penumbra[t - s.t_start, local(pen)] = True
            present[t - s.t_start] = True
        return umbra.reshape(n_w, h, w), penumbra.reshape(n_w, h, w), present


def track_spots(cube_cont: np.ndarray, time_h: np.ndarray, ds_name: str = 'DS',
                params: TrackingParams | None = None) -> TrackResult:
    """Detect, follow and filter every spot in `cube_cont`. See the module docstring."""
    params = params or TrackingParams()
    tracker = _Tracker(cube_cont.shape[1:], time_h, params)
    for t in range(cube_cont.shape[0]):
        tracker.step(t, cube_cont[t])

    mapping = _rename(tracker.spots)
    def rename(n):
        # A healed spot no longer exists: anything that pointed at it points at the spot
        # that absorbed it. '~' marks a name that left no frames for any other reason.
        while n and n not in mapping and tracker.spots[n].end_reason.startswith('healed_into:'):
            n = tracker.spots[n].end_reason.partition(':')[2]
        return mapping.get(n, f'~{n}') if n else n

    spots = {mapping[n]: s for n, s in tracker.spots.items() if s.frames}
    for s in spots.values():
        reason, _, other = s.end_reason.partition(':')
        if other:
            s.end_reason = f'{reason}:{rename(other)}'
    for e in tracker.events:
        # The event's own subject keeps its provisional '~' name when it left no frames, so
        # a healed split still reads as itself rather than as the spot it rejoined.
        e['spot'] = mapping.get(e['spot'], f"~{e['spot']}")
        e['other'] = rename(e['other'])
    for s in spots.values():
        if s.active:
            s.end_reason = 'end_of_data'
    rows = []
    for s in spots.values():
        area = np.array([len(u) + len(pen) for u, pen in s.frames.values()])
        umb_area = np.array([len(u) for u, _ in s.frames.values()])
        duration = float(time_h[s.t_end] - time_h[s.t_start])
        if s.parent is not None and s.penumbra_split:
            if np.median(area) < params.min_spot_px:
                s.kept, s.reject_reason = False, f'median area {np.median(area):.0f} px'
            elif duration < params.min_duration_h:
                s.kept, s.reject_reason = False, f'duration {duration:.1f} h'
        rows.append(dict(
            spot=display_name(ds_name, s), dir=dir_name(ds_name, s), parent=s.parent or '',
            t_start=s.t_start, t_end=s.t_end, time_start_h=float(time_h[s.t_start]),
            time_end_h=float(time_h[s.t_end]), duration_h=duration, n_frames=len(s.frames),
            median_area_px=float(np.median(area)), max_area_px=int(area.max()),
            median_umbra_px=float(np.median(umb_area)), penumbra_split=s.penumbra_split,
            end_reason=s.end_reason, kept=s.kept, reject_reason=s.reject_reason))
    events = pd.DataFrame(tracker.events, columns=['frame', 'event', 'spot', 'other', 'detail'])
    if len(events):
        events.insert(1, 'time_h', time_h[events['frame'].to_numpy()])
    table = pd.DataFrame(rows).sort_values(['t_start', 'spot'], ignore_index=True)
    return TrackResult(ds_name, cube_cont.shape, spots, events, table,
                       np.asarray(tracker.untracked), params)


def spot_data(data: dict, result: TrackResult, name: str,
              mag_filter: Callable[[np.ndarray], np.ndarray] | None = DS0N_DEFAULTS['mag_filter'],
              ) -> dict:
    """`load_ds0n_region`'s dict narrowed to one spot: cropped to its lifetime and to its
    bounding box (`TrackResult.bbox`).

    The spatial crop is what keeps memory bounded — full-box masks for one DS02 spot are
    ~0.45 GB each, and `compute_metrics` adds a full-size `|B|` copy on top; a whole-box run
    of DS02 was OOM-killed at 18 GB. It also centres the figures and the animation on the
    spot. The quiet-sun cubes are separate patches and are cropped in time only.

    The cubes are views, not copies. ``present['cont']`` marks the frames inside the window
    in which the spot had masks, so `compute_metrics` reports a NaN area — not 0 px — for a
    gap frame or one where the tracker briefly missed it.
    """
    s = result.spots[name]
    window = slice(s.t_start, s.t_end + 1)
    ys, xs = result.bbox(name)
    umbra, penumbra, present = result.masks(name, (ys, xs))
    both = umbra | penumbra
    out = {key: data[key][window, ys, xs] for key in ('cube_cont', 'cube_mag', 'cube_dop')}
    out.update({key: data[key][window] for key in ('cube_dop_qsun', 'cube_mag_qsun')})
    hot_spot = None
    if mag_filter is not None:
        with np.errstate(invalid='ignore'):
            hot_spot = mag_filter(out['cube_mag']) & both
    out.update(
        umbra=umbra, penumbra=penumbra, both=both, hot_spot=hot_spot,
        cube_mu=None if data.get('cube_mu') is None else data['cube_mu'][window, ys, xs],
        time_h=data['time_h'][window], n_t=umbra.shape[0], cadence_s=data['cadence_s'],
        umbra_thresh=result.params.umbra_thresh, penumbra_thresh=result.params.penumbra_thresh,
        present={'cont': present}, spot_name=display_name(result.ds_name, s),
        frame_offset=s.t_start, bbox=(ys, xs),
    )
    return out


def save_tracking_outputs(result: TrackResult, out_dir) -> None:
    """`spots.csv`, `events.csv` and the masks themselves as `masks.npz`.

    The masks are stored sparsely — per spot, the (frame, flat pixel index) pairs of its
    umbra and of its penumbra — because a shared penumbra belongs to two spots at once and
    so cannot go into one label cube. `TrackResult.masks` rebuilds the dense form.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(out_dir / 'spots.csv', index=False)
    result.events.to_csv(out_dir / 'events.csv', index=False)
    arrays = {'shape': np.array(result.shape)}
    for s in result.spots.values():
        for k, part in (('umbra', 0), ('penumbra', 1)):
            pairs = [np.column_stack([np.full(len(v[part]), t), v[part]])
                     for t, v in sorted(s.frames.items())]
            arrays[f'{dir_name(result.ds_name, s)}_{k}'] = np.concatenate(pairs).astype(np.int32)
    np.savez_compressed(out_dir / 'masks.npz', **arrays)


def plot_tracking_overview(cube_cont: np.ndarray, result: TrackResult, time_h: np.ndarray,
                           plots_dir=None, save: bool = True, n_panels: int = 4):
    """Continuum at a few frames with every kept spot outlined and named, plus a lifeline
    per spot. The one figure to look at before believing any per-spot number.

    Solid outline = umbra, dashed = outer edge of umbra + penumbra (spots sharing a penumbra
    draw the same dashed line). Lifelines of rejected spots are grey.
    """
    import matplotlib.pyplot as plt

    valid = np.flatnonzero(np.isfinite(cube_cont).any(axis=(1, 2)))
    frames = valid[np.linspace(0, len(valid) - 1, n_panels).round().astype(int)]
    names = sorted(result.spots, key=lambda n: (len(n), n))
    # tab10, not BOX_COLORS: those are for dark full-disk maps and include white, which
    # vanishes on the white lifetime panel.
    palette = plt.get_cmap('tab10').colors
    colors = {n: palette[i % len(palette)] for i, n in enumerate(names)}
    _, ny, nx = result.shape

    image_h = 4.5 * max(ny / nx, 0.4)
    fig = plt.figure(figsize=(4.5 * n_panels, image_h + 0.25 * len(names) + 2))
    grid = fig.add_gridspec(2, n_panels, height_ratios=[image_h, 0.25 * len(names) + 1])
    for i, t in enumerate(frames):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(cube_cont[t], cmap='gray', origin='lower')
        for s in result.kept():
            if t not in s.frames:
                continue
            u, pen = s.frames[t]
            for idx, style in ((u, '-'), (np.concatenate([u, pen]), '--')):
                if idx.size:
                    mask = np.zeros(ny * nx, dtype=bool)
                    mask[idx] = True
                    # Outer outline only: a shared penumbra excludes the other spot's
                    # umbra, which would otherwise draw as a hole inside this footprint.
                    mask = ndimage.binary_fill_holes(mask.reshape(ny, nx))
                    ax.contour(mask, levels=[0.5], colors=colors[s.name],
                               linestyles=style, linewidths=1.0)
            y, x = _centroid(u if u.size else pen, nx)
            ax.text(x, y, display_name('', s)[1:], color=colors[s.name], fontsize=11,
                    fontweight='bold', ha='center', va='center')
        ax.set_title(f'frame {t}  ·  {time_h[t]:.1f} h')
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(grid[1, :])
    for row, name in enumerate(names):
        s = result.spots[name]
        ts = np.array(sorted(s.frames))
        color = colors[name] if s.kept else '0.7'
        ax.plot(time_h[ts], np.full(ts.size, row), '|', color=color, markersize=8)
        label = display_name('', s)[1:] + ('' if s.kept else f'  (rejected: {s.reject_reason})')
        ax.text(time_h[ts[-1]], row, '  ' + label, va='center', fontsize=9)
    ax.set_yticks([]); ax.set_ylim(-1, len(names))
    ax.set_xlabel('time (h)')
    ax.set_title(f'{result.ds_name} — spot lifetimes (max {result.untracked.max(initial=0)} '
                 f'untracked spot-sized components in a frame)')
    fig.tight_layout()
    save_figure(fig, plots_dir, 'tracking_overview.png', save)
    plt.close(fig)
    return fig


def run_spot_pipeline(sdata: dict, out_dir, use_absolute_mag: bool = True) -> dict:
    """`03B` steps 2–6 on one spot, everything saved under `out_dir`, nothing shown."""
    import matplotlib.pyplot as plt

    from . import analysis, animation, spectra

    out_dir = pathlib.Path(out_dir)
    plots_dir = out_dir / 'plots'
    show, plt.show = plt.show, (lambda *a, **k: None)
    try:
        mid = sdata['n_t'] // 2
        analysis.plot_calibration_frame(sdata, frame_idx=mid, save=True, plots_dir=plots_dir)
        analysis.plot_magnetogram_masks(sdata, frame_idx=mid, save=True, plots_dir=plots_dir)

        metrics = analysis.compute_metrics(sdata, use_absolute_mag=use_absolute_mag)
        metrics = analysis.add_mag_residuals(sdata, metrics)
        analysis.save_metrics_csv(sdata, metrics, out_dir)

        analysis.plot_time_series(sdata, metrics, save=True, plots_dir=plots_dir,
                                  do_not_show_quiet=True)
        analysis.plot_time_series(sdata, metrics, normalized=True, save=True,
                                  plots_dir=plots_dir)
        analysis.plot_area(sdata, metrics, save=True, plots_dir=plots_dir)

        cadence_s, mag_variant = sdata['cadence_s'], metrics['mag_variant']
        psd = spectra.region_spectra(metrics, cadence_s, kind='psd', time_h=sdata['time_h'])
        psd_mag = spectra.region_spectra(metrics, cadence_s, kind='psd',
                                         key=f'mean_mag{mag_variant}_residual',
                                         time_h=sdata['time_h'])
        spectra.plot_spectra_combined(psd[0:3], cadence_s, kind='psd', xlim=(0, 0.1),
                                      log_y=False, save=True, plots_dir=plots_dir,
                                      print_table=False)
        spectra.plot_spectra_compare(psd, psd_mag, cadence_s, label_a='Dopplergram',
                                     label_b='Magnetogram residual', kind='psd',
                                     normalize=True, xlim=(0, 0.1), save=True,
                                     plots_dir=plots_dir, print_table=False)

        animation.save_animation(sdata, metrics, out_dir / 'anim.html',
                                 step=max(1, sdata['n_t'] // 60), fps=5,
                                 mag_symmetric_cbar=True)
    finally:
        plt.show = show
        plt.close('all')
    return metrics


def process_region(ds_dir, out_root, params: TrackingParams | None = None,
                   run_pipeline: bool = True) -> TrackResult:
    """One raw DS0N directory end to end: load, track, save, and run `03B` on every kept spot.

    Writes ``out_root/<DS>/`` (spots.csv, events.csv, masks.npz, plots/tracking_overview.png)
    and one ``out_root/<DS>_<spot>/`` per kept spot laid out exactly like a `03B` output
    directory, so `report.build_ds0n_summary(out_root)` picks them up unchanged. A spot whose
    pipeline fails is recorded in ``spots.csv`` (``pipeline_error``) rather than stopping the
    region.
    """
    import shutil
    import traceback

    from .loaders import load_ds0n_region, verify_cadence

    ds_dir = pathlib.Path(ds_dir)
    out_root = pathlib.Path(out_root)
    params = params or TrackingParams()

    _, gaps = verify_cadence(ds_dir, expected_s=DS0N_DEFAULTS['cadence_s'])
    # filter_mask=False: the loader's own masks are not used, the tracker builds its own.
    data = load_ds0n_region(ds_dir, umbra_thresh=params.umbra_thresh,
                            penumbra_thresh=params.penumbra_thresh, cadence_s=gaps,
                            filter_mask=False, mag_filter=None, normalize_mag_by_mu=True,
                            drop_out_of_order=True)
    result = track_spots(data['cube_cont'], data['time_h'], ds_dir.name, params)

    ds_out = out_root / ds_dir.name
    plot_tracking_overview(data['cube_cont'], result, data['time_h'], plots_dir=ds_out / 'plots')
    overview = ds_out / 'plots' / 'tracking_overview.png'

    errors = {}
    if run_pipeline:
        for s in result.kept():
            spot_dir = out_root / dir_name(ds_dir.name, s)
            print(f'── {display_name(ds_dir.name, s)}  ({len(s.frames)} frames)')
            try:
                run_spot_pipeline(spot_data(data, result, s.name), spot_dir)
                shutil.copy(overview, spot_dir / 'plots' / overview.name)
            except Exception as exc:                 # noqa: BLE001 — record, keep going
                errors[display_name(ds_dir.name, s)] = f'{type(exc).__name__}: {exc}'
                traceback.print_exc()
    result.table['pipeline_error'] = result.table['spot'].map(errors).fillna('')
    save_tracking_outputs(result, ds_out)
    return result
