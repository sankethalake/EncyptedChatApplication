import pickle
from bson.binary import Binary
from datetime import datetime
from keras.models import load_model
from bson.json_util import dumps
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_socketio import SocketIO, join_room, leave_room
from pymongo.errors import DuplicateKeyError
from helper import *
from db import get_user, save_user, save_room, add_room_members, get_rooms_for_user, get_room, is_room_member, \
    get_room_members, is_room_admin, update_room, remove_room_members, save_message, get_messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


UPLOAD_FOLDER = './received/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "sfdjkafnk"
socketio = SocketIO(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

alice = load_model('./models/alice.h5')
bob = load_model('./models/bob.h5')
eve = load_model('./models/eve.h5')
block_padding = 11
block_size = 16


@app.route('/about', methods=['GET', 'POST'])
def hello():
    cryp = []
    if request.method == 'POST':
        raw_message = request.form['alice_input']
        # print(raw_message)
        messages = processRawMessage(raw_message)
        message = messages[0]
        key = messages[1]
        print(request.url)
        cipher = alice.predict([message, key])
        decipher = (bob.predict([cipher, key]) > 0.5).astype(int)
        adversary = (eve.predict(cipher) > 0.5).astype(int)

        plaintext = processBinaryMessage(decipher)
        adv = processBinaryMessage(adversary)

        file = request.files['file']
        if 'file' not in request.files or file.filename == '':
            flash('No file part')
            cryp = [plaintext, adv, cipher]
            return render_template('about.html', cryp=cryp)
        else:
            # if user does not select file, browser also
            # submit a empty part without filename
            # if file.filename == '':
            #     # flash('No selected file')
            #     print(request.url)
            #     return redirect(request.url)
            file.save('./static/uploads/' + file.filename)
            enc_path, key, superkey = encryptImage(
                './static/uploads/' + file.filename)
            dec_img, adv_img = decryptImage(enc_path, key, superkey)

            cryp = [plaintext, adv, cipher, dec_img,
                    adv_img, '/uploads/' + file.filename]

    return render_template('about.html', cryp=cryp)


@app.route('/', methods=['GET', 'POST'])
def home():
    rooms = []
    if current_user.is_authenticated:
        rooms = get_rooms_for_user(current_user.username)
    return render_template("index.html", rooms=rooms)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    message = ''
    if request.method == 'POST':
        username = request.form.get('username')
        password_input = request.form.get('password')
        user = get_user(username)

        if user and user.check_password(password_input):
            login_user(user)
            return redirect(url_for('home'))
        else:
            message = 'Failed to login!'
    return render_template('login.html', message=message)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    message = ''
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        try:
            save_user(username, email, password)
            return redirect(url_for('login'))
        except DuplicateKeyError:
            message = "User already exists!"
    return render_template('signup.html', message=message)


@app.route("/logout/")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/create-room/', methods=['GET', 'POST'])
@login_required
def create_room():
    message = ''
    if request.method == 'POST':
        room_name = request.form.get('room_name')
        usernames = [username.strip() for username in request.form.get('members').split(',')]

        if len(room_name) and len(usernames):
            room_id = save_room(room_name, current_user.username)
            if current_user.username in usernames:
                usernames.remove(current_user.username)
            add_room_members(room_id, room_name, usernames, current_user.username)
            return redirect(url_for('view_room', room_id=room_id))
        else:
            message = "Failed to create room"
    return render_template('create_room.html', message=message)


@app.route('/rooms/<room_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_room(room_id):
    room = get_room(room_id)
    if room and is_room_admin(room_id, current_user.username):
        existing_room_members = [member['_id']['username']
                                 for member in get_room_members(room_id)]
        room_members_str = ",".join(existing_room_members)
        message = ''
        if request.method == 'POST':
            room_name = request.form.get('room_name')
            room['name'] = room_name
            update_room(room_id, room_name)

            new_members = [username.strip()
                           for username in request.form.get('members').split(',')]
            members_to_add = list(
                set(new_members) - set(existing_room_members))
            members_to_remove = list(
                set(existing_room_members) - set(new_members))
            if len(members_to_add):
                add_room_members(room_id, room_name,
                                 members_to_add, current_user.username)
            if len(members_to_remove):
                remove_room_members(room_id, members_to_remove)
            message = 'Room edited successfully'
            room_members_str = ",".join(new_members)
        return render_template('edit_room.html', room=room, room_members_str=room_members_str, message=message)
    else:
        return "Room not found", 404


@app.route('/rooms/<room_id>/')
@login_required
def view_room(room_id):
    room = get_room(room_id)
    if room and is_room_member(room_id, current_user.username):
        room_members = get_room_members(room_id)
        messages, received_message = get_messages(room_id)
        return render_template('view_room.html', username=current_user.username, room=room, room_members=room_members,
                               messages=messages)
    else:
        return "Room not found", 404


@app.route('/rooms/<room_id>/messages/')
@login_required
def get_older_messages(room_id):
    room = get_room(room_id)
    if room and is_room_member(room_id, current_user.username):
        page = int(request.args.get('page', 0))
        messages, received_message = get_messages(room_id, page)
        return dumps(messages)
    else:
        return "Room not found", 404


@socketio.on('send_message')
def handle_send_message_event(data):
    app.logger.info("{} has sent message to the room {}: {}".format(data['username'],
                                                                    data['room'],
                                                                    data['message']))
    data['created_at'] = datetime.now().strftime("%d %b, %H:%M")
    cipher, key = textEncryption(data['message'])
    eve_dec = eveTextDecryption(cipher)
    print(eve_dec)
    # data['eve'] = Binary(pickle.dumps(eve_dec, protocol=2), subtype=128)
    data['eve'] = eve_dec
    # record['feature2'] = Binary(pickle.dumps(npArray, protocol=2), subtype=128 )
    data['cipher'] = Binary(pickle.dumps(cipher, protocol=2), subtype=128)
    data['key'] = Binary(pickle.dumps(key, protocol=2), subtype=128)
    # print(data)
    # print(data['cipher'])
    save_message(data['room'], data['message'],
                 data['username'], data['cipher'], data['key'], data['eve'])
    socketio.emit('receive_message', data, room=data['room'])


@socketio.on('join_room')
def handle_join_room_event(data):
    app.logger.info("{} has joined the room {}".format(
        data['username'], data['room']))
    join_room(data['room'])
    socketio.emit('join_room_announcement', data, room=data['room'])


@socketio.on('leave_room')
def handle_leave_room_event(data):
    app.logger.info("{} has left the room {}".format(
        data['username'], data['room']))
    leave_room(data['room'])
    socketio.emit('leave_room_announcement', data, room=data['room'])


@login_manager.user_loader
def load_user(username):
    return get_user(username)


if __name__ == '__main__':
    socketio.run(app, debug=True)
