import json
from copy import deepcopy
from getpass import getpass
from typing import Optional

from dflow.context import Context
from dflow.executor import Executor, render_script_with_tmp_root
from dflow.op_template import PythonScriptOPTemplate, ShellOPTemplate
from dflow.utils import randstr
from dflow.workflow import Workflow


succ_code = [0, "0000"]

class LebesgueContext(Context):
    """
    Lebesgue context

    Args:
        username: user name for Lebesgue
        password: password for Lebesgue
        login_url: login url for Lebesgue
        app_name: application name
        org_id: organization ID
        user_id: user ID
        tag: tag
        executor: executor
        extra: extra arguments
        authorization: JWT token
    """

    def __init__(
            self,
            username: Optional[str] = None,
            password: Optional[str] = None,
            lbg_url: str = "https://lbg-api.dp.tech",
            login_url: str = "https://workflow.dp.tech/account_gw/login",
            app_name: Optional[str] = None,
            org_id: Optional[str] = None,
            user_id: Optional[str] = None,
            tag: Optional[str] = None,
            executor: Optional[str] = None,
            extra: Optional[dict] = None,
            authorization: Optional[str] = None,
    ) -> None:
        self.lbg_url = lbg_url
        self.login_url = login_url
        self.username = username
        self.password = password
        self.app_name = app_name
        self.org_id = org_id
        self.user_id = user_id
        self.tag = tag
        self.executor = executor
        self.extra = extra
        self.authorization = authorization
        if not self.app_name == "Hermite":
            self.login()

    def login(self):
        if self.authorization is None:
            if self.username is None:
                self.username = input("Lebesgue username: ")
            if self.password is None:
                self.password = getpass("Lebesgue password: ")
            data = {
                "username": self.username,
                "password": self.password,
            }
            import requests
            rsp = requests.post(self.login_url, headers={
                                "Content-type": "application/json"}, json=data)
            res = json.loads(rsp.text)
            if res["code"] not in succ_code:
                if "error" in res:
                    raise RuntimeError("Login failed: %s" %
                                       res["error"]["msg"])
                elif "message" in res:
                    raise RuntimeError("Login failed: %s" % res["message"])
                else:
                    raise RuntimeError("Login failed")
            self.authorization = res["data"]["token"]

    def render(self, template):
        if isinstance(template, Workflow):
            template.annotations["workflow.dp.tech/app_name"] = self.app_name
            template.annotations["workflow.dp.tech/org_id"] = self.org_id
            template.annotations["workflow.dp.tech/user_id"] = self.user_id
            template.annotations["workflow.dp.tech/tag"] = self.tag
            template.annotations["workflow.dp.tech/executor"] = self.executor
            template.annotations["task.dp.tech/extra"] = json.dumps(
                self.extra) if isinstance(self.extra, dict) else self.extra
            template.annotations["workflow.dp.tech/authorization"] = \
                self.authorization
            template.annotations["workflow.dp.tech/executor_addr"] = \
                self.lbg_url + "/"
            return template

        if isinstance(template, (ShellOPTemplate, PythonScriptOPTemplate)):
            new_template = deepcopy(template)
            new_template.name += "-" + randstr()
            new_template.annotations["workflow.dp.tech/executor"] = \
                self.executor
            if self.executor == "lebesgue_v2":
                new_template.script = render_script_with_tmp_root(template,
                                                                  "$(pwd)/tmp")
                if isinstance(template, ShellOPTemplate):
                    new_template.script = "mkdir -p tmp\n" + \
                        new_template.script
                if isinstance(template, PythonScriptOPTemplate):
                    new_template.script = "import os\nos.makedirs('tmp', "\
                        "exist_ok=True)\n" + new_template.script
            return new_template

        return template


class LebesgueExecutor(Executor):
    """
    Lebesgue executor

    Args:
        executor: executor
        extra: extra arguments, will override extra defined in global context
    """

    def __init__(
            self,
            executor: Optional[str] = None,
            extra: Optional[dict] = None,
    ) -> None:
        self.executor = executor
        self.extra = extra

    def render(self, template):
        new_template = deepcopy(template)
        new_template.name += "-" + randstr()
        if self.executor is not None:
            new_template.annotations["workflow.dp.tech/executor"] = \
                self.executor
        if self.extra is not None:
            new_template.annotations["task.dp.tech/extra"] = json.dumps(
                self.extra) if isinstance(self.extra, dict) else self.extra
        if self.executor == "lebesgue_v2":
            new_template.script = render_script_with_tmp_root(template,
                                                              "$(pwd)/tmp")
            if isinstance(template, ShellOPTemplate):
                new_template.script = "mkdir -p tmp\n" + new_template.script
            if isinstance(template, PythonScriptOPTemplate):
                new_template.script = "import os\nos.makedirs('tmp', "\
                    "exist_ok=True)\n" + new_template.script
        return new_template