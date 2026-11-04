import logging

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def report_device():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        logging.info("[设备] %s SM %s", torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device))
    logging.info("[配置] PyTorch %s，测试 FP32 CUDA 接口", torch.__version__)
    logging.info("\n[阶段] Python 接口数值正确性验证")


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 0:
        logging.info("[SUCCESS] Python 测试通过")
