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


import shlex
import subprocess
import datetime
import time

from converter import logging


def run(cmd_string, work_dir=None, timeout=None, is_shell=True, log_level="debug"):
    """
         执行一个SHELL命令 封装了subprocess的Popen方法, 支持超时判断，支持读取stdout和stderr
        :parameter:
              cwd: 运行命令时更改路径，如果被设定，子进程会直接先更改当前路径到cwd
              timeout: 超时时间，秒，支持小数，精度0.1秒
              shell: 是否通过shell运行
        :return return_code
        :exception 执行超时
    """

    if is_shell:
        cmd_string_list = cmd_string
    else:
        cmd_string_list = shlex.split(cmd_string)
    if timeout:
        end_time = datetime.datetime.now() + \
                   datetime.timedelta(seconds=timeout)

    sub = subprocess.Popen(cmd_string_list,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           shell=True,
                           bufsize=0,
                           cwd=work_dir,
                           close_fds=True)
    while True:
        line = sub.stdout.readline().decode('utf-8')
        if log_level == "error":
            logging.error(str(line))
        elif log_level == "debug":
            logging.debug(str(line))
        if line == '' and sub.poll() is not None:
            break
    rc = sub.poll()
    return rc
