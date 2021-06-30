import config
import logging
import os

from aiogram import Bot, Dispatcher, executor, types
import pandas as pd
from NN import NN

# log
logging.basicConfig(level = logging.INFO)

nn = NN()

nn.predict()

bot = Bot(token = "**********")
dp = Dispatcher(bot)

@dp.message_handler(commands=["start"], commands_prefix="/")
async def start(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ["/start", "/help", "/show_stocks", "/show_best", "/show_worst", "/show_prediction"]
    keyboard.add(*buttons)
    await message.answer(
"""Hi! It's char-bot with invest recommendations
You can write next commands for getting info:
/start - it will write you introduction message
/show_stocks - it will show you all stock
/show_best - it will show you best stock
/show_worst - it will show you worst stock
/show_prediction - it will show you sorted prediction number for every stock
/help - get help message""", reply_markup=keyboard
    )

@dp.message_handler(commands=["help"], commands_prefix="/")
async def help(message: types.Message):
    await message.answer(
"""/start - it will write you introduction message
/show_stocks - it will show you all stock
/show_best - it will show you best stock
/show_worst - it will show you worst stock
/show_prediction - it will show you sorted prediction number for every stock
/help - get help message"""
    )

@dp.message_handler(commands=["show_stocks"], commands_prefix="/")
async def show_stocks(message: types.Message):
    await message.answer(nn.get_stocks())

@dp.message_handler(commands=["show_best"], commands_prefix="/")
async def show_best(message: types.Message):
    await message.answer(nn.get_best())

@dp.message_handler(commands=["show_worst"], commands_prefix="/")
async def show_worst(message: types.Message):
    await message.answer(nn.get_worst())

@dp.message_handler(commands=["show_prediction"], commands_prefix="/")
async def show_prediction(message: types.Message):
    await message.answer(nn.get_prediction())

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=False)
