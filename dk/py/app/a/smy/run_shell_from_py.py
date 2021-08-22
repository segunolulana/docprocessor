import os
import subprocess
import logging

# Set logging to display useful information, will be used later to have a flag like -vvv to set verbosity
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

new_env = os.environ.copy()
new_env["qss"] = os.environ["HOME"] + "/Tasks/Summarizer/dk/py/app/a/smy"


def run_command_raw_n_shell(cmd, cwd, stdout=None, stderr=None):
    '''Execute a command output going to stdout/stderr'''
    log.info("Executing '%s' in directory %s", subprocess.list2cmdline(cmd), cwd)
    p = subprocess.Popen(
        cmd, cwd=cwd, stdout=stdout, stderr=stderr, shell=True, env=new_env, universal_newlines=True, bufsize=1
    )
    while p.poll() is None:
        if p.stdout:
            output = p.stdout.readline()
            if output:
                print(output)
    # p.wait()
    # stdout, stderr = p.communicate()
    # assert p.returncode == 0, stderr


CWD = os.environ["HOME"]
run_command_raw_n_shell(
    "python $qss/run_sum.py --kw g -r -f '~/Dropbox/orgnotes/mygtd*.org' -m luhn -a --sm s --lg --dw --no", CWD
)
