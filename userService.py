import dbService as c


def login(username, password):
    if username == "" or username is None or password == "" or password is None:
        return False
    # add action for login here
    data = c.login_get_user(username, password)
    if len(data) > 0:
        return True


def register(username, password, fullname):
    if username == "" or username is None or password == "" or password is None or fullname == "" or fullname is None:
        return False
    result = c.register(username, password, fullname)

    if result is not None or result is not 0:
        return True
