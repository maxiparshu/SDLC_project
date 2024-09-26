
import kivy
import random

kivy.require('1.9.0')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

class MyWindow(BoxLayout):
    def generate_number(self):
        self.ids.random_label.text = str(random.randint(0, 2000))

class AndroidApp(App):
    def build(self):
        return MyWindow()

if __name__ == '__main__':
    AndroidApp().run()