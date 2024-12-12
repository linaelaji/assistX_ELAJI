import os
import tornado.ioloop
import tornado.web
import tornado.websocket
import json
from helper import query_with_groq_api, load_data, generate_embeddings, load_embeddings, search_similar_documents
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("templates/index.html")


class ChatHandler(tornado.websocket.WebSocketHandler):
    def initialize(self, df, embeddings, conversation_history, executor):
        self.df = df
        self.embeddings = embeddings
        self.conversation_history = conversation_history
        self.executor = executor

    def open(self):
        print("WebSocket opened")

    async def on_message(self, message):
        print("start on message")
        data = json.loads(message)
        user_input = data['message']
        model = data.get('model', 'groq')

        self.conversation_history.append(
            {'role': 'client', 'content': user_input})

        if user_input:
            print(user_input, model)
            print("Start embedding the question")
            similar_docs = search_similar_documents(
                self.df, user_input, self.embeddings)
            self.write_message(json.dumps(
                {"type": "docs", "documents": similar_docs}))

            if model == "groq":
                response = query_with_groq_api(user_input, os.getenv(
                    'GROQ_API_KEY'), self.conversation_history, similar_docs)

                self.write_message({"content": response, "is_complete": True})
                self.conversation_history.append(
                    {'role': 'proxigen', 'content': response, 'is_complete': True})

            elif model == 'ollama':
                print('ollama implementation')

    def on_close(self):
        print("WebSocket closed")


class ResetHandler(tornado.web.RequestHandler):
    def initialize(self, conversation_history):
        self.conversation_history = conversation_history

    def post(self):
        self.conversation_history.clear()
        self.write(json.dumps({"status": "Conversation history reset"}))


def make_app():
    df = load_data()
    print(df.head())
    if not os.path.exists('data/embeddings.npy'):
        print("Generating embeddings for knowledge base...")
        generate_embeddings(df)
    embeddings = load_embeddings()

    conversation_history = []
    executor = ThreadPoolExecutor()

    static_path = os.path.join(os.path.dirname(__file__), "static")

    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/chat", ChatHandler, dict(df=df, embeddings=embeddings,
         conversation_history=conversation_history, executor=executor)),
        (r"/reset", ResetHandler, dict(conversation_history=conversation_history)),
    ],
        static_path=static_path,
        cookie_secret="DATAX",
        debug=True,
        compiled_template_cache=False)


if __name__ == "__main__":
    app = make_app()
    app.listen(8888, address='0.0.0.0')
    tornado.ioloop.IOLoop.current().start()
