import time
import threading
import hydra
import cv2 as cv
import tkinter as tk
import numpy as np

from PIL import Image, ImageTk
from functools import partial
from copy import copy
from typing import List, Dict


def instantiate_by_name(config: List[Dict], object_name: str):
    for obj in config:
        if obj.params.name == object_name:
            return hydra.utils.instantiate(obj)
    raise KeyError(f"{object_name} not found in config")


def make_default_params(config: List[Dict], object_name: str):
    for obj in config:
        if obj.params.name == object_name:
            return {k: v.default for k, v in obj.params.items() if k != 'name'}

    raise KeyError(f"{object_name} not found in config")


class CameraApp:
    def __init__(self, config, window: tk.Tk, window_name: str = "My Camera",
                 window_width: int = 1200, window_height: int = 400, window_bg: str = 'black',
                 buttons_width: int = 20, buttons_height: int = 2, buttons_font_size: int = 15):

        self.config = config
        self.pipeline_steps = config.pipeline_steps
        self.pipeline = {k: {'name': 'identity', 'func': lambda x: x, 'default_func': lambda x: x} for k in self.pipeline_steps}

        self.window = window
        self.window_width = window_width
        self.window_height = window_height
        self.ImageLabel = tk.Label(self.window, width=int(self.window_width * 5 / 6), height=self.window_height,
                                   bg="black")
        self.set_window_params(window_name, window_width, window_height, window_bg)

        self.buttons_width = buttons_width
        self.buttons_height = buttons_height
        self.buttons_font_size = buttons_font_size

        self.buttons = self.create_buttons()
        self.main()

    @staticmethod
    def hide_sliders(button):
        if button.get('params') is not None:
            for param_name_value in button['params'].values():
                param_name_value['label'].place_forget()
                param_name_value['slider'].place_forget()

    def show_sliders(self, button):
        if button.get('params') is not None:
            for param_name_value in button['params'].values():
                param_name_value['label'].place(x=param_name_value['x'],
                                                y=param_name_value['y'] - 22 * self.buttons_height,
                                                anchor='center')
                param_name_value['slider'].place(x=param_name_value['x'], y=param_name_value['y'], anchor='center')
                param_name_value['slider'].set(param_name_value['default'])

    def update_pipeline(self, step, enhancer):
        button = self.buttons[step][enhancer]
        button['state'] = np.logical_not(button['state'])
        if button['state']:
            self.show_sliders(button)
        else:
            self.hide_sliders(button)

        if self.pipeline[step]['name'] == enhancer:
            self.pipeline[step]['name'] = 'identity'
            self.pipeline[step]['default_func'] = lambda x: x
            self.pipeline[step]['func'] = lambda x: x
        else:
            if self.pipeline[step]['name'] != 'identity':
                self.buttons[step][self.pipeline[step]['name']]['state'] = False
                self.hide_sliders(self.buttons[step][self.pipeline[step]['name']])
            self.pipeline[step]['name'] = enhancer
            self.pipeline[step]['func'] = lambda x: instantiate_by_name(self.config[step], enhancer).enhance_image(x, **make_default_params(self.config[step], enhancer))

    def reset_pipelines_params(self, value, step, enhancer, param_name):
        params = {param_name: float(value)}
        self.pipeline[step]['func'] = lambda x: instantiate_by_name(self.config[step], enhancer).enhance_image(x, **params)

    def set_window_params(self, window_name, window_width, window_height, window_bg):
        self.window.title(window_name)
        self.window.geometry(f"{window_width}x{window_height}")
        self.window.configure(bg=window_bg)
        self.window.resizable(1, 1)
        self.ImageLabel.place(x=0, y=0)

    def create_buttons(self):
        buttons = {step: {} for step in self.pipeline_steps}
        x, y = int(self.window_width * 11 / 12), int(self.buttons_height * 10)
        for step in self.pipeline_steps:
            tk.Label(self.window, text=" ".join(step.upper().split('_')), font=("Times", 4*self.buttons_font_size),
                     bg="black", relief='flat').place(x=x, y=y, anchor='center')
            y += int(self.buttons_height * 30)

            for enhancer in self.config.get(step):
                buttons[step][enhancer.params.name] = \
                    {'button': tk.Button(self.window, width=self.buttons_width, height=self.buttons_height,
                                         text=enhancer.params.name, font=("Times", self.buttons_font_size),
                                         bg="#2F4F4F", relief='flat',
                                         command=partial(self.update_pipeline, step=step, enhancer=enhancer.params.name)),
                     'state': False
                     }
                buttons[step][enhancer.params.name]['button'].place(x=x, y=y, anchor='center')

                if len(enhancer.params) > 1:
                    buttons[step][enhancer.params.name]['params'] = {}
                    for i, (k, v) in enumerate(enhancer.params.items()):
                        if k != 'name':
                            buttons[step][enhancer.params.name]['params'][k] = {
                                'label': tk.Label(self.window, text=k, font=("Times", self.buttons_font_size),
                                                  bg="black", relief='flat'),
                                'slider': tk.Scale(self.window, from_=v.range[0], to=v.range[1],
                                                   length=30*self.buttons_height,
                                                   resolution=v.step,
                                                   command=partial(self.reset_pipelines_params, step=step,
                                                                   enhancer=enhancer.params.name, param_name=k)),
                                'default': v.default,
                                'x': int(self.window_width + 3*(i-1)*self.buttons_width),
                                'y': y
                            }
                            buttons[step][enhancer.params.name]['params'][k]['slider'].set(v.default)

                y += int(self.buttons_height * 40)

        return buttons

    @staticmethod
    def load_camera():
        camera = cv.VideoCapture(0)
        if camera.isOpened():
            ret, frame = camera.read()
        while ret:
            ret, frame = camera.read()
            if ret:
                yield frame
            else:
                yield False

    def main(self):
        self.render_thread = threading.Thread(target=self.start_camera)
        self.render_thread.daemon = True
        self.render_thread.start()

    def start_camera(self):
        frame = self.load_camera()
        while True:
            Frame = next(frame)
            if frame:
                picture = cv.cvtColor(Frame, cv.COLOR_BGR2RGB)
                picture_2 = copy(picture)
                for step in self.pipeline_steps:
                    picture_2 = self.pipeline[step]['func'](picture_2)

                picture = Image.fromarray(np.hstack([picture, picture_2]))
                picture = picture.resize((1000, 400), resample=0)
                picture = ImageTk.PhotoImage(picture)
                self.ImageLabel.configure(image=picture)
                self.ImageLabel.photo = picture
                time.sleep(0.001)


@hydra.main(config_path="configs", config_name="camera_app_setup")
def my_app(config):
    root = tk.Tk()
    App = CameraApp(config, root)
    root.mainloop()


if __name__ == "__main__":
    my_app()

