from __future__ import division
def calculator(input_str):
    return eval(input_str)

if __name__ == "__main__":
    while True:
        try:
            input_expression = input()
            # print("[Calculator({})->{}]".format(input_expression, calculator(input_expression)))
        except:
            break