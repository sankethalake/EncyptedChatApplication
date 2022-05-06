from email.mime import image
import pickle
import numpy as np
import base64
from PIL import Image
import random
import string
from keras.models import load_model

# TO BE KEPT CONSTANT THROUGHOUT THE PROJECT
block_size_unpadded = 5
block_padding = 11
block_size = 16
chrlist = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z', '.', ',', '!', '?',
    ':', ' '
]
alice = load_model('./models/alice.h5')
bob = load_model('./models/bob.h5')
eve = load_model('./models/eve.h5')
binlist = [
    '00000', '00001', '00010', '00011', '00100',
    '00101', '00110', '00111', '01000', '01001',
    '01010', '01011', '01100', '01101', '01110',
    '01111', '10000', '10001', '10010', '10011',
    '10100', '10101', '10110', '10111', '11000',
    '11001', '11010', '11011', '11100', '11101',
    '11110', '11111'
]

# To generate random bits for padding


def randombits(n):
    if n == 0:
        return ''
    decvalue = np.random.randint(0, 2**n)
    formatstring = '0' + str(n) + 'b'
    return format(decvalue, formatstring)

# To convert a character string into binary string


def encstr(message, block_padding=0):
    cipher = ''
    bintext = ' '.join('{0:016b}'.format(ord(x), 'b') for x in message)
    cipher = bintext.replace(" ", "")

    return [cipher, len(message)]

# To convert binary cipher text into deiphered text


def decstr(cipher, n, block_padding=0):
    seperated = ''
    for i in range(len(cipher)):
        if i % 16 == 0:
            seperated += " "
        seperated += cipher[i]

    # print(seperated)
    bin_list = seperated.split()
    text = ''

    for bin in bin_list:
        an_integer = int(bin, 2)
        ascii_character = chr(an_integer)
        text += ascii_character

    return text

# Convert a string of binary characters into a corresponding numpy array


def strToArr(bin_string, block_size):
    bin_list = []
    keys = []
    letter_count = 0
    innerList = []

    for letter in bin_string:
        innerList.append(int(letter))
        letter_count += 1
        if (letter_count % block_size) == 0:
            bin_list.append(innerList)
            innerList = []
            key_bit = np.random.randint(0, 2, 16)
            keys.append(key_bit)

    input_list = np.array(bin_list)
    key_list = np.array(keys)
    return [input_list, key_list]

# To convert a binary numpy array to a string of binary characters


def arrToStr(bin_arr):
    bin_string = ''

    for inner in bin_arr:
        for bit in inner:
            bin_string += str(bit)
    return bin_string

# Combines above functions into one to encrypt a text message


def processRawMessage(raw_message):
    encrypt = encstr(raw_message, block_padding)
    bin_cipher = strToArr(encrypt[0], block_size)
    bin_message = bin_cipher[0]
    bin_key = bin_cipher[1]

    return [bin_message, bin_key]

# Combines above functions into one to decrypt a text message


def processBinaryMessage(binary_message):
    message_str = arrToStr(binary_message)
    decipher = decstr(message_str, len(binary_message), block_padding)
    return decipher

# Converts an image list to a string capable of encryption


def processImageList(image):
    converted_string = base64.b64encode(image)
    return converted_string

# Error counter


def testEquality(original, deciphered):
    count = 0
    if len(original) != len(deciphered):
        return -1
    else:
        for i in range(len(original)):
            if original[i] != deciphered[i]:
                count += 1
    return count


def getImageMatrix(imageName):
    im = Image.open(imageName)
    pix = im.load()
    color = True
    if type(pix[0, 0]) == int:
        color = False
    image_size = im.size
    image_matrix = []
    for width in range(int(image_size[0])):
        row = []
        for height in range(int(image_size[1])):
            row.append((pix[width, height]))
        image_matrix.append(row)
    return image_matrix, image_size[0], image_size[1], color


def getImageMatrix_gray(imageName):
    im = Image.open(imageName).convert('LA')
    pix = im.load()
    image_size = im.size
    image_matrix = []
    color = 1
    if type(pix[0, 0]) == int:
        color = 0
    for width in range(int(image_size[0])):
        row = []
        for height in range(int(image_size[1])):
            row.append((pix[width, height]))
        image_matrix.append(row)
    return image_matrix, image_size[0], color


def LogisticEncryption(imageName, key):
    N = 256
    key_list = [ord(x) for x in key]
    G = [key_list[0:4], key_list[4:8], key_list[8:12]]
    g = []
    R = 1
    for i in range(1, 4):
        s = 0
        for j in range(1, 5):
            s += G[i-1][j-1] * (10**(-j))
        g.append(s)
        R = (R*s) % 1

    L = (R + key_list[12]/256) % 1
    S_x = round(((g[0]+g[1]+g[2])*(10**4) + L * (10**4)) % 256)
    V1 = sum(key_list)
    V2 = key_list[0]
    for i in range(1, 13):
        V2 = V2 ^ key_list[i]
    V = V2/V1

    L_y = (V+key_list[12]/256) % 1
    S_y = round((V+V2+L_y*10**4) % 256)
    C1_0 = S_x
    C2_0 = S_y
    C = round((L*L_y*10**4) % 256)
    C_r = round((L*L_y*10**4) % 256)
    C_g = round((L*L_y*10**4) % 256)
    C_b = round((L*L_y*10**4) % 256)
    x = 4*(S_x)*(1-S_x)
    y = 4*(S_y)*(1-S_y)

    imageMatrix, dimensionX, dimensionY, color = getImageMatrix(imageName)
    print(color)
    LogisticEncryptionIm = []
    for i in range(dimensionX):
        row = []
        for j in range(dimensionY):
            while x < 0.8 and x > 0.2:
                x = 4*x*(1-x)
            while y < 0.8 and y > 0.2:
                y = 4*y*(1-y)
            x_round = round((x*(10**4)) % 256)
            y_round = round((y*(10**4)) % 256)
            C1 = x_round ^ ((key_list[0]+x_round) %
                            N) ^ ((C1_0 + key_list[1]) % N)
            C2 = x_round ^ ((key_list[2]+y_round) %
                            N) ^ ((C2_0 + key_list[3]) % N)
            if color:
                C_r = ((key_list[4]+C1) % N) ^ ((key_list[5]+C2) % N) ^ (
                    (key_list[6]+imageMatrix[i][j][0]) % N) ^ ((C_r + key_list[7]) % N)
                C_g = ((key_list[4]+C1) % N) ^ ((key_list[5]+C2) % N) ^ (
                    (key_list[6]+imageMatrix[i][j][1]) % N) ^ ((C_g + key_list[7]) % N)
                C_b = ((key_list[4]+C1) % N) ^ ((key_list[5]+C2) % N) ^ (
                    (key_list[6]+imageMatrix[i][j][2]) % N) ^ ((C_b + key_list[7]) % N)
                row.append((C_r, C_g, C_b))
                C = C_r

            else:
                C = ((key_list[4]+C1) % N) ^ ((key_list[5]+C2) %
                                              N) ^ ((key_list[6]+imageMatrix[i][j]) % N) ^ ((C + key_list[7]) % N)
                row.append(C)

            x = (x + C/256 + key_list[8]/256 + key_list[9]/256) % 1
            y = (x + C/256 + key_list[8]/256 + key_list[9]/256) % 1
            for ki in range(12):
                key_list[ki] = (key_list[ki] + key_list[12]) % 256
                key_list[12] = key_list[12] ^ key_list[ki]
        LogisticEncryptionIm.append(row)

    im = Image.new("L", (dimensionX, dimensionY))
    if color:
        im = Image.new("RGB", (dimensionX, dimensionY))
    else:
        # L is for Black and white pixels
        im = Image.new("L", (dimensionX, dimensionY))

    pix = im.load()
    for x in range(dimensionX):
        for y in range(dimensionY):
            pix[x, y] = LogisticEncryptionIm[x][y]
    im.save(imageName[:-4] + "_LogisticEnc.png", "PNG")
    return imageName[:-4] + "_LogisticEnc.png"


def LogisticDecryption(imageName, key, decrypterName):
    N = 256
    key_list = [ord(x) for x in key]

    G = [key_list[0:4], key_list[4:8], key_list[8:12]]
    g = []
    R = 1
    for i in range(1, 4):
        s = 0
        for j in range(1, 5):
            s += G[i-1][j-1] * (10**(-j))
        g.append(s)
        R = (R*s) % 1

    L_x = (R + key_list[12]/256) % 1
    S_x = round(((g[0]+g[1]+g[2])*(10**4) + L_x * (10**4)) % 256)
    V1 = sum(key_list)
    V2 = key_list[0]
    for i in range(1, 13):
        V2 = V2 ^ key_list[i]
    V = V2/V1

    L_y = (V+key_list[12]/256) % 1
    S_y = round((V+V2+L_y*10**4) % 256)
    C1_0 = S_x
    C2_0 = S_y

    C = round((L_x*L_y*10**4) % 256)
    I_prev = C
    I_prev_r = C
    I_prev_g = C
    I_prev_b = C
    I = C
    I_r = C
    I_g = C
    I_b = C
    x_prev = 4*(S_x)*(1-S_x)
    y_prev = 4*(L_x)*(1-S_y)
    x = x_prev
    y = y_prev
    imageMatrix, dimensionX, dimensionY, color = getImageMatrix(imageName)

    henonDecryptedImage = []
    for i in range(dimensionX):
        row = []
        for j in range(dimensionY):
            while x < 0.8 and x > 0.2:
                x = 4*x*(1-x)
            while y < 0.8 and y > 0.2:
                y = 4*y*(1-y)
            x_round = round((x*(10**4)) % 256)
            y_round = round((y*(10**4)) % 256)
            C1 = x_round ^ ((key_list[0]+x_round) %
                            N) ^ ((C1_0 + key_list[1]) % N)
            C2 = x_round ^ ((key_list[2]+y_round) %
                            N) ^ ((C2_0 + key_list[3]) % N)
            if color:
                I_r = ((((key_list[4]+C1) % N) ^ ((key_list[5]+C2) % N) ^ (
                    (I_prev_r + key_list[7]) % N) ^ imageMatrix[i][j][0]) + N-key_list[6]) % N
                I_g = ((((key_list[4]+C1) % N) ^ ((key_list[5]+C2) % N) ^ (
                    (I_prev_g + key_list[7]) % N) ^ imageMatrix[i][j][1]) + N-key_list[6]) % N
                I_b = ((((key_list[4]+C1) % N) ^ ((key_list[5]+C2) % N) ^ (
                    (I_prev_b + key_list[7]) % N) ^ imageMatrix[i][j][2]) + N-key_list[6]) % N
                I_prev_r = imageMatrix[i][j][0]
                I_prev_g = imageMatrix[i][j][1]
                I_prev_b = imageMatrix[i][j][2]
                row.append((I_r, I_g, I_b))
                x = (x + imageMatrix[i][j][0]/256 +
                     key_list[8]/256 + key_list[9]/256) % 1
                y = (x + imageMatrix[i][j][0]/256 +
                     key_list[8]/256 + key_list[9]/256) % 1
            else:
                I = ((((key_list[4]+C1) % N) ^ ((key_list[5]+C2) % N) ^
                     ((I_prev+key_list[7]) % N) ^ imageMatrix[i][j]) + N-key_list[6]) % N
                I_prev = imageMatrix[i][j]
                row.append(I)
                x = (x + imageMatrix[i][j]/256 +
                     key_list[8]/256 + key_list[9]/256) % 1
                y = (x + imageMatrix[i][j]/256 +
                     key_list[8]/256 + key_list[9]/256) % 1
            for ki in range(12):
                key_list[ki] = (key_list[ki] + key_list[12]) % 256
                key_list[12] = key_list[12] ^ key_list[ki]
        henonDecryptedImage.append(row)
    if color:
        im = Image.new("RGB", (dimensionX, dimensionY))
    else:
        # L is for Black and white pixels
        im = Image.new("L", (dimensionX, dimensionY))
    pix = im.load()
    for x in range(dimensionX):
        for y in range(dimensionY):
            pix[x, y] = henonDecryptedImage[x][y]
    im.save(imageName.split('_')[0] +
            "_LogisticDec_" + decrypterName + ".png", "PNG")
    return imageName.split('_')[0] + "_LogisticDec_" + decrypterName + ".png"


def encryptImage(imageName):
    key = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))
    enc_path = LogisticEncryption(imageName, key)
    messages = processRawMessage(key)
    message = messages[0]
    superkey = messages[1]

    cipher = alice.predict([message, superkey])

    return enc_path, cipher, superkey


def decryptImage(imageName, cipher, superkey):
    bob_pred = (bob.predict([cipher, superkey]) > 0.5).astype(int)
    eve_pred = (eve.predict(cipher) > 0.5).astype(int)

    key = processBinaryMessage(bob_pred)
    adv = processBinaryMessage(eve_pred)

    dec_image = LogisticDecryption(imageName, key, "bob")
    adv_image = LogisticDecryption(imageName, adv, "eve")

    return dec_image[8:], adv_image[8:]


def textEncryption(raw_message):
    messages = processRawMessage(raw_message)
    '''Do not remove tolist(). Used to convert numpy array to json compatible format. To undo use np.array(data['cipher'])'''
    message = messages[0]
    key = messages[1]
    cipher = alice.predict([message, key])
    return cipher, key


def textDecryption(jsonCipher, jsonKey):
    cipher = pickle.loads(jsonCipher)
    key = pickle.loads(jsonKey)
    print(type(key))

    decipher = (bob.predict([cipher, key]) > 0.5).astype(int)
    plaintext = processBinaryMessage(decipher)
    return plaintext


def eveTextDecryption(cipher):
    decipher = (eve.predict(cipher) > 0.5).astype(int)
    adv = processBinaryMessage(decipher)
    return adv
