import logging
import os
import sys
import random
import re
import spacy
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, CallbackQueryHandler, ContextTypes
from dotenv import load_dotenv
from enum import Enum
from spacy.tokens import Doc
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from textblob import TextBlob

BOT_TOKEN = '7469206121:AAHGCAXqsVGJt-hyn08YmRIjQlSWB-8Q0xg'

class State(Enum):
    BANTER = 0,
    START = 1,
    BUTTON_HANDLER = 2,
    VOCAB_HANDLER = 3,
    TRANSLATE_HANDLER = 4,
    LANGUAGE_SELECT = 5,
    ANALYZE_HANDLER = 6


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

load_dotenv()


class SentenceTyper(spacy.matcher.Matcher):
    def __init__(self, vocab):
        super().__init__(vocab)
        self.add("WH-QUESTION", [[{"IS_SENT_START": True, "TAG": {"IN": ["WDT", "WP", "WP$", "WRB"]}}]])
        self.add("YN-QUESTION",
                 [[{"IS_SENT_START": True, "TAG": "MD"}, {"POS": {"IN": ["PRON", "PROPN", "DET"]}}],
                  [{"IS_SENT_START": True, "POS": "VERB"}, {"POS": {"IN": ["PRON", "PROPN", "DET"]}}, {"POS": "VERB"}]])
        self.add("INSTRUCTION",
                 [[{"IS_SENT_START": True, "TAG": "VB"}],
                  [{"IS_SENT_START": True, "LOWER": {"IN": ["please", "kindly"]}}, {"TAG": "VB"}]])
        self.add("WISH",
                 [[{"IS_SENT_START": True, "TAG": "PRP"}, {"TAG": "MD"},
                   {"POS": "VERB", "LEMMA": {"IN": ["love", "like", "appreciate"]}}],
                  [{"IS_SENT_START": True, "TAG": "PRP"},
                   {"POS": "VERB", "LEMMA": {"IN": ["want", "need", "require"]}}]])
        self.add("EXCLAMATORY", [
            [{"IS_SENT_START": True, "LOWER": {"IN": ["wow", "amazing", "incredible", "fantastic", "great", "oh"]}}]
        ])
        self.add("DECLARATIVE", [
            [{"IS_SENT_START": True, "TAG": {"IN": ["NN", "NNS", "NNP", "NNPS"]}},
             {"POS": {"IN": ["AUX", "VERB"]}}, {"POS": {"IN": ["NOUN", "ADJ", "ADV", "PRON"]}}]
        ])

    def __call__(self, *args, **kwargs):
        matches = super().__call__(*args, **kwargs)
        if matches:
            match_id, _, _ = matches[0]
            if match_id == self.vocab["WH-QUESTION"]:
                return wh_question_handler
            elif match_id == self.vocab["YN-QUESTION"]:
                return yn_question_handler
            elif match_id == self.vocab["WISH"]:
                return wish_handler
            elif match_id == self.vocab["INSTRUCTION"]:
                return instruction_handler
        else:
            return generic_handler
        if len(matches) > 1:
            logger.debug(f"NOTE: SentenceTyper actually found {len(matches)} matches.")


class VerbFinder(spacy.matcher.DependencyMatcher):
    def __init__(self, vocab):
        super().__init__(vocab)
        self.add("VERBPHRASE", [
            [{"RIGHT_ID": "root", "RIGHT_ATTRS": {"DEP": "ROOT"}},
             {"LEFT_ID": "root", "REL_OP": ">", "RIGHT_ID": "auxiliary", "RIGHT_ATTRS": {"TAG": "VB"}},
             {"LEFT_ID": "root", "REL_OP": ">", "RIGHT_ID": "modal", "RIGHT_ATTRS": {"TAG": "MD"}}],
            [{"RIGHT_ID": "root", "RIGHT_ATTRS": {"DEP": "ROOT"}},
             {"LEFT_ID": "root", "REL_OP": ">", "RIGHT_ID": "auxiliary", "RIGHT_ATTRS": {"POS": "AUX"}}],
            [{"RIGHT_ID": "root", "RIGHT_ATTRS": {"DEP": "ROOT"}}]
        ])

    def __call__(self, *args, **kwargs):
        verbmatches = super().__call__(*args, **kwargs)
        if verbmatches:
            if len(verbmatches) > 1:
                logging.debug(f"NOTE: VerbFinder actually found {len(verbmatches)} matches.")
                for verbmatch in verbmatches:
                    logging.debug(verbmatch)
            _, token_idxs = verbmatches[0]
            return sorted(token_idxs)


povs = {
    "I am": "you are",
    "I was": "you were",
    "I'm": "you're",
    "I'd": "you'd",
    "I've": "you've",
    "I'll": "you'll",
    "you are": "I am",
    "you were": "I was",
    "you're": "I'm",
    "you'd": "I'd",
    "you've": "I've",
    "you'll": "I'll",
    "I": "you",
    "my": "your",
    "your": "my",
    "yours": "mine",
    "you": "I",
    "me": "you",
}
povs_c = re.compile(r'\b({})\b'.format('|'.join(re.escape(pov) for pov in povs)))


def generate_reply(sentence: Doc, verbs_idxs: list, prefix: list, suffixes: list) -> str:
    # Initialize reply with the first word of the sentence
    reply = [sentence[0].text.lower()]

    # Extract the subject (nsubj)
    subject = next((chunk.text for chunk in sentence.noun_chunks if chunk.root.dep_ == 'nsubj'), None)
    if subject:
        reply.append(subject)

    # Append verbs
    verbs = " ".join([sentence[i].text.lower() for i in verbs_idxs])
    reply.append(verbs)

    # Extract the direct object (dobj)
    direct_object = next((chunk.text for chunk in sentence.noun_chunks if chunk.root.dep_ == 'dobj'), None)
    if direct_object:
        reply.append(direct_object)

    # Join the reply parts into a single string
    reply_str = " ".join(reply)

    # Replace pronouns or other specific words using regex
    reply_str = re.sub(povs_c, lambda match: povs.get(match.group(), match.group()), reply_str)

    # Add a random prefix and suffix
    reply_str = random.choice(prefix) + reply_str + random.choice(suffixes)

    return reply_str


def wh_question_handler(nlp, sentence, verbs_idxs):
    return generate_reply(
        sentence, verbs_idxs,
        prefix=["Good question! ", "Hmm... ", "Let me think... "],
        suffixes=[
            ", but I'm not sure right now.",
            ". I might have to do some digging.",
            ", but let me get back to you on that."
        ]
    )


def yn_question_handler(nlp, sentence, verbs_idxs):
    return generate_reply(
        sentence, verbs_idxs,
        prefix=["You know, ", "Well, "],
        suffixes=[
            " I wish I could tell you for sure.",
            ". Let's keep it a mystery for now.",
            " I'll have to get back to you on that one."
        ]
    )


def wish_handler(nlp, sentence, verbs_idxs):
    reply = sentence.text
    reply = re.sub(povs_c, lambda match: povs.get(match.group()), reply)
    return random.choice([
        "Noted! ", "Got it! "
    ]) + reply + random.choice([
        " I'll see what I can do.",
        " Let's make it happen."
    ])


def instruction_handler(nlp, sentence, verbs_idxs):
    reply = sentence.text
    reply = re.sub(povs_c, lambda match: povs.get(match.group()), reply)
    return random.choice([
        "Sure thing: ", "Alright: "
    ]) + reply + random.choice([
        " Let's get this done!",
        " I'm on it."
    ])


def generic_handler(nlp, sentence, verbs_idxs):
    logging.debug("INVOKING GENERIC HANDLER")
    reply = sentence.text
    reply = re.sub(povs_c, lambda match: povs.get(match.group()), reply)
    return reply


def load_small_talk_responses(file_path):
    small_talk_responses = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, responses = line.strip().split(':')
            small_talk_responses[key] = responses.split(';')
    return small_talk_responses


# Load the small talk responses from the text file
small_talk_responses = load_small_talk_responses('small_talk_responses.txt')


async def banter(update: Update,  context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text.lower()

    # Check if the user's message matches any of the small talk responses
    for key, responses in small_talk_responses.items():
        if key in user_message:
            reply = random.choice(responses)
            await update.message.reply_text(reply)
            return

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(update.message.text)
    sentencetyper = SentenceTyper(nlp.vocab)
    verbfinder = VerbFinder(nlp.vocab)

    reply = ''
    for sentence in doc.sents:
        verbs_idxs = verbfinder(sentence.as_doc())
        reply += (sentencetyper(sentence.as_doc()))(nlp, sentence, verbs_idxs)

    await update.message.reply_text(reply)
    return

# Translates the user's input based on a simple dictionary.
translate_dict = {
    "hello": {"es": "hola", "fr": "bonjour", "de": "hallo", "xo": "molo"},
    "thank you": {"es": "gracias", "fr": "merci", "de": "danke", "xo": "enkosi"},
    "goodbye": {"es": "adiós", "fr": "au revoir", "de": "auf Wiedersehen", "xo": "sala kakuhle"},
    "apple": {"es": "manzana", "fr": "pomme", "de": "apfel", "xo": "apile"},
    "car": {"es": "coche", "fr": "voiture", "de": "auto", "xo": "imoto"},
    "run": {"es": "correr", "fr": "courir", "de": "laufen", "xo": "baleka"},
}


def load_vocab_words(file_path):
    vocab_words = {}
    with open(file_path, 'r') as file:
        for line in file:
            word, definition = line.strip().split(':')
            vocab_words[word] = definition
    return vocab_words


vocab_words = load_vocab_words('vocab_words.txt')

async def start(update, context):
    user_name = update.message.from_user.first_name
    keyboard = [
        [InlineKeyboardButton("Practice Vocabulary", callback_data='vocab')],
        [InlineKeyboardButton("Translate Text", callback_data='translate')],
        [InlineKeyboardButton("Analyze Sentence", callback_data='analyze')],
        [InlineKeyboardButton("hmu, with feedback thanks❤️", url='https://t.me/AmbeMajavu')]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        f"Hey {user_name}! 🎉 Dive into the Language Learning Bot and pick an option below to get started! 🚀"
        ,
        reply_markup=reply_markup,
    )
    return State.BUTTON_HANDLER


async def interact(update, context):
    await update.message.reply_text("Hi there! I'm here to chat with you. What would you like to talk about?")
    return State.BANTER


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == 'vocab':
        word = random.choice(list(vocab_words.keys()))
        context.user_data['vocab_word'] = word
        await query.edit_message_text(f"What does '{word}' mean?")
        return State.VOCAB_HANDLER

    elif query.data == 'translate':
        # Prompt the user to select a language
        keyboard = [
            [InlineKeyboardButton("Spanish", callback_data='es')],
            [InlineKeyboardButton("French", callback_data='fr')],
            [InlineKeyboardButton("German", callback_data='de')],
            [InlineKeyboardButton("Xhosa", callback_data='xo')],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text("Please select the target language for translation:", reply_markup=reply_markup)
        return State.LANGUAGE_SELECT

    elif query.data == 'analyze':
        await query.edit_message_text("Send a sentence for me to analyze! 😊")
        return State.ANALYZE_HANDLER

    return State.BANTER


async def language_select_handler(update, context):
    query = update.callback_query
    await query.answer()

    context.user_data['target_language'] = query.data
    await query.edit_message_text(f"Type in the word/phrase you want translated 🌟{query.data.upper()}.")
    return State.TRANSLATE_HANDLER


async def translate_message_handler(update: Update, context):
    # Tokenize the user's input
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(update.message.text.lower())
    # Initialize a list to hold the translated words
    translated_text = []
    missing_words = []

    # Get the target language from context
    target_language = context.user_data.get('target_language', 'es')  # Default to Spanish if not set

    # Iterate over each word in the user's input
    for token in doc:
        # Check if the word exists in the translation dictionary
        if token.text in translate_dict:
            translated_word = translate_dict[token.text].get(target_language, token.text)
            translated_text.append(translated_word)
        else:
            # If the word does not exist in the dictionary, add it to the missing words list
            missing_words.append(token.text)
            translated_text.append(token.text)

    # Join the translated words into a single string
    translated_text_str = " ".join(translated_text)

    # Perform sentiment analysis
    sentiment = TextBlob(update.message.text).sentiment
    sentiment_response = "Type another word or phrase for me to translate! 🚀 " if sentiment.polarity > 0 else "Let’s do this, give me a another one!😄"

    # If there are missing words, notify the user
    if missing_words:
        missing_words_str = ", ".join(missing_words)
        await update.message.reply_text(
            f"Oooops, I couldn't find translations for the following words: {missing_words_str}.\n\n"
            "Got another word/phrase for me? 🧐"
            "Need a break, just hit /cancel.")
    else:
        await update.message.reply_text(f"Translated text: {translated_text_str}\n\n{sentiment_response}")


async def vocab_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'vocab_word' in context.user_data:
        word = context.user_data['vocab_word']
        user_answer = update.message.text.lower()

        # Check if the user's answer matches the English definition
        correct_answer = vocab_words[word].lower()
        if user_answer == correct_answer:
            await update.message.reply_text("Correct! 🎉")
        else:
            await update.message.reply_text(f"Sorry, the correct answer is: {correct_answer}\n\n"
                                            "Need a break? Just hit /cancel to wrap things up 💎")

        # Ask another random question about a different word
        next_word = random.choice(list(vocab_words.keys()))
        context.user_data['vocab_word'] = next_word
        question_formats = [
            f"What is the meaning of '{next_word}'?",
            f"Can you define '{next_word}'?",
            f"How would you describe '{next_word}'?",
            f"What does '{next_word}' refer to?",
        ]
        await update.message.reply_text(random.choice(question_formats))


async def analyze_message_handler(update: Update, context):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(update.message.text)
    analysis = "\n".join([f"{token.text}: {token.pos_}" for token in doc])

    # Perform sentiment analysis
    sentiment = TextBlob(update.message.text).sentiment
    sentiment_response = "Got another sentence for me? Let's dive in!😎" if sentiment.polarity > 0 else "Keep them coming!🔥 "

    await update.message.reply_text(f"Sentence Analysis:\n{analysis}\n\n{sentiment_response}\n\n"
                                    "Need a break? Just hit /cancel to wrap things up 🚀", )


async def cancel(update, context):
    await update.message.reply_text(
        "Thanks for the chat. I'll be off then! If you want to start again, just type or click /start.", )
    return ConversationHandler.END


async def help_command(update, context):
    await update.message.reply_text(
        "Here are some commands you can use:\n"
        "/interact - chat with your language coach\n"
        "/start - Explore learning activities\n"
        "/cancel - End the current session\n\n"
        "Feel free to try out any of these commands!",
    )
    return ConversationHandler.END


# Function to handle errors
async def error(update, context):
    logger.warning(f'Update {update} caused error {context.error}')


def main():
    print("Starting bot....")
    token = os.getenv(BOT_TOKEN)
    application = Application.builder().token(BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler(['start'], start),
                      CommandHandler(['interact'], interact)],
        states={
            State.BANTER: [MessageHandler(filters.TEXT & ~filters.COMMAND, banter)],
            State.BUTTON_HANDLER: [CallbackQueryHandler(button_handler)],
            State.LANGUAGE_SELECT: [CallbackQueryHandler(language_select_handler)],
            State.TRANSLATE_HANDLER: [MessageHandler(filters.TEXT & ~filters.COMMAND, translate_message_handler)],
            State.VOCAB_HANDLER: [MessageHandler(filters.TEXT & ~filters.COMMAND, vocab_message_handler)],
            State.ANALYZE_HANDLER: [MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_message_handler)]
        },
        fallbacks=[CommandHandler('interact', interact),
                   CommandHandler('start', start),
                   CommandHandler(['cancel'], cancel),
                   CommandHandler('help', help_command)]
    )
    # Add handlers to application
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('interact', interact))
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('cancel', cancel))
    application.add_handler(CommandHandler('help', help_command))

    # Error handler
    application.add_error_handler(error)

    print("Polling...")
    application.run_polling(poll_interval=3)


if __name__ == '__main__':
    main()
