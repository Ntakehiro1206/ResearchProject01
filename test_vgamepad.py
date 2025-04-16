import vgamepad as vg
from pynput import keyboard

gamepad = vg.VX360Gamepad()

def on_press(key):
    try:
        if key == keyboard.Key.space:
            print("Aボタンを押す")
            gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            gamepad.update()
    except AttributeError:
        pass

def on_release(key):
    try:
        if key == keyboard.Key.space:
            print("Aボタンを離す")
            gamepad.release_button()
            gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            gamepad.update()
    except AttributeError:
        pass

with keyboard.Listener(on_press=on_press, on_release=on_press) as listener:
    print("スペースキーでAボタンを制御中... [Ctrl + Cで終了]")
    listener.join()