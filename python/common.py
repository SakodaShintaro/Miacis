def seconds_to_pretty_str(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    remain = seconds - (hours * 3600)
    minutes = remain // 60
    seconds = remain - (minutes * 60)
    days = hours // 24
    hours = hours % 24
    return f'{days:02}:{hours:02}:{minutes:02}:{seconds:02}'
