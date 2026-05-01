============================================================
  VERIFICATION: uncalibrated env | 50 episodes | 10000 steps
  Model: models/checkpoints/dqn_uncalibrated/dqn_best.pth
============================================================

  ACTION DISTRIBUTION:
    edge_1  :  1658 ( 16.6%) |████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| ✓
    edge_2  :  1277 ( 12.8%) |██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| ✓
    cloud   :  1643 ( 16.4%) |████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| ✓
    reject  :  5422 ( 54.2%) |███████████████████████████░░░░░░░░░░░░░░░░░░░░░░░| ✓

  PERFORMANCE:
    Avg reward     : -61.57
    Avg latency    : 83.3ms
    SLA rate       : 44.7%
    Avg cost       : 0.5239

  ✅ DIVERSE — action distribution tốt, sẵn sàng deploy.

  [DQNAgent] Model loaded ← models/checkpoints/dqn_calibrated/dqn_best.pth

============================================================
  VERIFICATION: calibrated env | 50 episodes | 10000 steps
  Model: models/checkpoints/dqn_calibrated/dqn_best.pth
============================================================

  ACTION DISTRIBUTION:
    edge_1  :  3086 ( 30.9%) |███████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| ✓
    edge_2  :   417 (  4.2%) |██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| ✓
    cloud   :  2020 ( 20.2%) |██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| ✓
    reject  :  4477 ( 44.8%) |██████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░| ✓

  PERFORMANCE:
    Avg reward     : -52.29
    Avg latency    : 7217.4ms
    SLA rate       : 52.1%
    Avg cost       : 0.6191

  ✅ DIVERSE — action distribution tốt, sẵn sàng deploy.