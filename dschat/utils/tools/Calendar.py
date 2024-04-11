import datetime
import calendar
def Calendar(input_str):
    now = datetime.datetime.now()
    return f'[Calendar()->{calendar.day_name[now.weekday()]}, {calendar.month_name[now.month]} {now.day}, {now.year}]'

if __name__ == "__main__":
    print(Calendar())