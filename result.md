EXECUTION_BACKEND=k8s python3 -m dispatcher.dispatcher_cli \
 --policy dqn --model models/checkpoints/dqn_calibrated/dqn_best.pth \
 --num-tasks 50 --concurrency 5 --seed 100 \
 --prometheus http://localhost:9090
kết quả:
03:57:28 [INFO] SmartDispatcher: Task task_000049 → cloud | backend=k8s job=task-task-000049-1777694234373 status=succeeded total_ms=14200 submit=0 startup=1000 exec=10000 poll=3139
CLI: [ 1/50] task_000001 -> rejected | lat= 0.0ms | SLA=MISS
CLI: [ 2/50] task_000002 -> cloud | lat=29789.0ms | SLA=OK
CLI: [ 3/50] task_000003 -> cloud | lat=18579.0ms | SLA=OK
CLI: [ 4/50] task_000004 -> cloud | lat=27648.0ms | SLA=MISS
CLI: [ 5/50] task_000005 -> cloud | lat=16397.0ms | SLA=OK
...
CLI: [ 47/50] task_000047 -> cloud | lat=9448.0ms | SLA=OK
CLI: [ 48/50] task_000048 -> cloud | lat=5208.0ms | SLA=OK
CLI: [ 49/50] task_000049 -> cloud | lat=14200.0ms | SLA=OK
CLI: [ 50/50] task_000050 -> rejected | lat= 0.0ms | SLA=MISS
=======================================================
Dispatcher Summary (50 tasks, policy=dqn)
SLA Rate : 52.0%
Avg Latency : 7266.9 ms
P95 Latency : 20267.8 ms
Avg Cost : 1.85202
Cloud Usage : 56.0%
Edge Usage : 44.0%
=======================================================
03:57:28 [INFO] CLI: Total time: 79.50s (0.6 tasks/sec)
