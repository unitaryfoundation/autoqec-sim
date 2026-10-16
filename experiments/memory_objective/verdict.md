## Verdict

Status: complete at smoke scale on the laptop (one to two threads).
Profiles: `ion` is the chain preset
(arXiv:2606.06455 rates, sequential gates) at scale 0.25, `board` the
qldpc-challenge recipe at `p = 1e-3`, `heron` the Heron r2 rates at scale
0.1; every tier is a leading-order or sampled estimate that needs the
expected number of faults per shot below about one, and the presets at
scale 1 exceed that on these circuits (chain 0.6 to 3.4, Heron 2.8 to
8.8, board 0.5 to 2.1).

1. Tiers and cost. `proxy` (closed form, weight-2 maximum-likelihood
   failure from the DEM plus exact `d_circ`) costs 0.08 to 0.22 s per code
   at `n <= 9` and 1.7 to 2.3 s for the [[16,4,4]] code, both bases,
   `R = d`, one thread; `sampled` (exact ML when `D + O <= 20`, else the
   weight-2 lookup decoder) 0.16 to 0.31 s and 5 s; `bposd` 0.2 to 1.5 s
   and 37 to 42 s at 200 failures per basis. Inside PPO the proxy
   evaluated 92 distinct target-distance codes at 0.47 s each.
2. Rank agreement. Proxy against the sampled ML tier: within 25 percent
   on every benchmark code below one expected fault per shot and Spearman
   0.9 on every profile over the five codes. Proxy against bposd: 0.9
   (ion), 0.8 (board), 0.3 (heron) on the five codes, the Heron outlier
   being Steane, for which bposd is 1.7 to 3.2 times worse than both ML
   tiers on every profile. On 32 random `[[9,1,3]]` codes (generator
   weights 3 to 9, per-round rates within a factor 2.5 of each other)
   proxy against bposd is 0.79 (ion), 0.90 (board), 0.78 (heron) at 100
   failures per basis; on the same codes two bposd seeds agree with each
   other only to 0.53 (ion), 0.95 (board), 0.88 (heron), so on the ion
   profile the proxy is closer to a single bposd sample than two bposd
   samples are to each other (at 400 failures per basis on ion: proxy
   against the sampled ML tier 0.93, bposd seed against seed 0.58), and
   the residual disagreement on board and heron (proxy against the
   two-seed mean 0.90 and 0.74) is within what the shot noise and bposd's
   own suboptimality allow. The proxy was not
   iterated beyond the hook-aware order and the leading-order weights;
   the one failure mode found is outside its regime (the routed non-CSS
   code under Heron at scale 1: proxy 1.8e-4, bposd 0.49).
3. Bias sensitivity. Under a Z-biased idle channel (every rate 2.5e-4,
   idle bias `eta`) the objective prefers XZZX over the plain rotated
   surface code at `n = 9`: ratio 1.00 at `eta = 0.5` (exact tie), 1.21 at
   2, 1.80 at 10, 2.27 at pure Z under sequential timing, bposd 1.08,
   1.23, 1.89, 2.19; parallel timing 1.00 to 2.10 (bposd to 2.35). Under
   the Forte set (`eta = 10` duration idle) the proxy and the near-ML
   lookup decoder prefer XZZX by 1.22 and 1.21; bposd does not (0.78).
   Under the chain set the two tie (0.99): its readout flips dominate.
   The encoder objective tied the two codes exactly under every profile.
4. Hooks and flags. The sorted gate order gives the Shor code `d_circ = 1`
   in the Z basis and the rotated surface code 2 in the X basis; the
   hook-aware order (default) gives 3 and 3. With bare ancillas every
   other benchmark code has `d_circ = 2`; one flag per generator lifts
   all of them to 3 and lowers the ion `p_round` 2.6 to 4.4 times, while
   under the board recipe the extra gates cost more than the flags gain.
5. Lattice. The `square:3x3` layout embeds the surface code and XZZX
   without routing; the non-CSS 3x3 code needs 63 extra two-qubit gates
   per round and 7 sequential batches, the Shor code 45 and 4, and both
   lose one unit of `d_circ` to the ladders' hooks.
6. Regression. PPO on the 3x3 lattice under the ion profile with the
   memory reward reaches distance 3 at 83k env steps (400k-step run, 62 s
   at one thread); the test-suite version (`R = 2`, 300k steps) passes.
