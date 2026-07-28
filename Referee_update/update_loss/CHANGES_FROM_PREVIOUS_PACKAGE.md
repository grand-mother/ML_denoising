# What changed relative to the previous package

The earlier package required more work than the referee requested. This replacement makes the following changes:

- wrapped-phase retraining is now conditional on a post-hoc audit;
- no magnitude-weighted phase loss is requested;
- no phase-weight scan or new Ray Tune search is requested;
- the existing time-only and dual-branch checkpoints are first evaluated on one common test sample rather than retrained automatically;
- a fixed split is required for common evaluation and for any conditional retraining, but the entire historical Ray search does not need to be rerun;
- the paper is corrected to match the production architecture and training code instead of changing code solely to match the draft;
- repository-wide SNR refactoring is not required; only publication scripts must use the reviewed definition;
- bootstrap intervals and per-bin counts are optional rather than mandatory;
- figures are regenerated only when a measured inconsistency affects a retained result.
