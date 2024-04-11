from dschat.utils.tools.Calculator import *
from dschat.utils.tools.Calendar import *

str_to_tool_api = {
    "Calculator": calculator,
    "Calendar": Calendar,
}

import re
PATTERN = re.compile(r'\[([a-zA-Z]+)\(([\w\s\(\)\+\-\*/,]*)\)\]')

def generate_tool_response(input_str):
    input_str = input_str.strip().replace(" ","")
    match = PATTERN.search(input_str)
    tool = match.group(1)
    param = match.group(2)
    if tool == "None":
        return "[None()]"
    if tool == "Calculator":
        param = param.replace(",", "")
    # print(tool, param)
    answer = str_to_tool_api[tool](param)
    if tool == "Calculator":
        answer = "{:.2f}".format(answer)
    return "[{}({})->{}]".format(tool, param, answer)

if __name__ == "__main__":
    while True:
        try:
            input_str = input("输入需要解析运算的API\n")
            print(generate_tool_response(input_str))
        except:
            break