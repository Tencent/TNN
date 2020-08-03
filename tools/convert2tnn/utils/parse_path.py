# Tencent is pleased to support the open source community by making TNN available.
#
# Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import os
import sys

from converter import logging
from utils import return_code


def parse_path(path: str):
    if path is None:
        return None
    if " " in path or " " in os.getcwd():
        logging.error("The path can not contain spaces!")
        sys.exit(return_code.SPACE_IN_PATH)
    if path.startswith("/"):
        return path
    elif path.startswith("./"):
        return os.path.join(os.getcwd(), path[2:])
    elif path.startswith("../"):
        abs_path = os.getcwd() + "/" + path
        return abs_path
    elif path.startswith("~"):
        abs_path = os.path.expanduser('~') + path[1:]
        return abs_path
    else:
        return os.path.join(os.getcwd(), path)
