"""
dispatcher package
==================
Smart Dispatcher cho hệ thống Hybrid Edge-Cloud.

Modules:
    - state_builder    : Xây dựng state vector từ Prometheus metrics
    - model_loader     : Load DQN/PPO model với hot-reload
    - error_handlers   : Retry logic, fallback, circuit breaker
    - smart_dispatcher : Core dispatcher logic
    - dispatcher_cli   : CLI interface
"""
