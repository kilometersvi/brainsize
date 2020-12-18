import requests
import io
import os

def get_content(index_str):
    try:
        link = "http://neurosciencelibrary.org/specimens/cetacea/dolphin/sections/coronal-cell/"+index_str+"tursiops66-127.jpg"

        response = requests.get(link)
        if str(response.content) == "b'The page cannot be displayed because an internal server error has occurred.'":
            return None
        return response.content
    except Exception as e:
        print("err: "+str(e))
        return None

def get_index_str(index):
    index_str = str(index)+"0"
    if index < 10:
        index_str = "00"+index_str
    elif index < 100:
        index_str = "0"+index_str
    return index_str

for i in range(0,56):
    index = (i+1)*5

    content = get_content(get_index_str(index))
    if content is None:
        for r in range(-4,5):
            if r == 0:
                continue
            content = get_content(get_index_str(index+r))
            if content is not None:
                index += r
                break
    if content is None:
        print("["+str(i)+"] image not found")
        continue

    print("["+str(i)+"] "+get_index_str(index))
    file = open("img/"+str(i)+"_dolphin_"+get_index_str(index)+".png", "wb")
    file.write(content)
    file.close()
