import sqlite3

db_connection = sqlite3.connect('data.db')
c = db_connection.cursor()


# db service function
def create_table_user():
    c.execute('CREATE TABLE users('
              'id integer PRIMARY KEY, '
              'username varchar,'
              'password varchar,'
              'password_hash TEXT,'
              'fullname varchar)')


def create_table_image():
    c.execute('CREATE TABLE images('
              'id integer PRIMARY KEY, '
              'image TEXT)')


def create_table_prediction():
    c.execute('CREATE TABLE prediction('
              'id integer PRIMARY KEY, '
              'img_prediction varchar, '
              'probability varchar, '
              'overall_prediction varchar,'
              'image_id INTEGER, '
              'CONSTRAINT fk_images FOREIGN KEY (image_id) REFERENCES images(id))')


def register(username, password, fullname, password_hash=None):
    c.execute(f'INSERT INTO users (username, password, password_hash, fullname) values("{username}", "{password}", "dummy","{fullname}")')
    db_connection.commit()
    return c.lastrowid


def login_get_user(username, password):
    c.execute(f'SELECT * FROM users WHERE username = "{username}" AND password = "{password}";')
    data = c.fetchall()
    return data


def get_image_by_id(id):
    c.execute(f'SELECT * FROM images where id={id}')
    data = c.fetchall()
    return data


def get_prediction_by_id(id):
    c.execute(f'SELECT * FROM prediction WHERE id={id}')
    data = c.fetchall()
    return data


def get_image_and_prediction_by_id(id):
    c.execute(f'SELECT * FROM prediction p INNER JOIN images i ON i.id = p.id WHERE p.id={id}')
    data = c.fetchall()
    return data


def get_image_and_prediction_all():
    c.execute('SELECT p.id, i.image, p.img_prediction, probability, overall_prediction, i.id '
              'FROM prediction p INNER JOIN images i ON i.id = p.id')
    data = c.fetchall()
    return data


def insert_image(image):
    c.execute(f'INSERT INTO images (image) VALUES("{image}")')
    db_connection.commit()
    return c.lastrowid


def insert_prediction(img_prediction, probability, overall_prediction, image_id):
    c.execute(f'INSERT INTO prediction ('
              f'img_prediction, probability, overall_prediction, image_id) '
              f'VALUES("{img_prediction}", "{probability}", "{overall_prediction}",{image_id})')
    db_connection.commit()
    return c.lastrowid


def delete_image(id):
    c.execute(f'DELETE FROM images WHERE id={id}')
    db_connection.commit()


def delete_prediction(id):
    c.execute(f'DELETE FROM prediction WHERE id={id}')
    db_connection.commit()


if __name__ == '__main__':
    create_table_user()
    create_table_prediction()
    create_table_image()


