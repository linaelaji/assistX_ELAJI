document
  .getElementById("chat-input")
  .addEventListener("input", autoResize, false);

function autoResize() {
  this.style.height = "auto";
  this.style.height = this.scrollHeight + "px";
}

document
  .getElementById("selected-model")
  .addEventListener("click", function () {
    document.getElementById("model-options").style.display =
      document.getElementById("model-options").style.display === "block"
        ? "none"
        : "block";
  });

var currentModel = "groq";

function updateModel(modelName) {
  let displayName;
  switch (modelName) {
    case "openai":
      displayName = "OpenAI GPT-4o";
      break;
    case "scaleway":
      displayName = "Scaleway LLM";
      break;
    case "groq":
      displayName = "Groq Llama-3 8B";
      break;
    default:
      displayName = modelName;
  }
  document.getElementById("selected-model").innerHTML =
    displayName + ' <span class="dropdown-arrow">&#9660;</span>';
  document.getElementById("model-options").style.display = "none";
  currentModel = modelName;
}

var ws = new WebSocket("ws://localhost:8888/chat");
var currentResponse = "";
var currentMessageElement = null;

ws.onopen = function (event) {
  console.log("Connection opened");
};

ws.onmessage = function (event) {
  var chatBox = document.getElementById("chat-box");
  var sendButton = document.getElementById("send-button");
  var data = JSON.parse(event.data);
  console.log(data);

  if (data.documents) {
    console.log("Top Documents and Their Scores:", data.documents);
  }

  if (data.content) {
    var newContent = currentResponse + data.content;
    if (!currentMessageElement) {
      currentMessageElement = document.createElement("div");
      currentMessageElement.innerHTML = formatMessage("proxigen", newContent);
      chatBox.appendChild(currentMessageElement);
    } else if (currentMessageElement.innerHTML.indexOf(data.content) === -1) {
      currentMessageElement.innerHTML = formatMessage("proxigen", newContent);
    }
    currentResponse = newContent;
    chatBox.scrollTop = chatBox.scrollHeight;
    var loadingElement = document.querySelector(".loading");
    if (loadingElement) loadingElement.remove();
    currentResponse = "";
    currentMessageElement = null;

    sendButton.classList.remove("disabled");
    sendButton.disabled = false;
  }
};

ws.onclose = function () {
  console.log("Connection closed");
  currentMessageElement = null;
};

function sendMessage() {
  var chatBox = document.getElementById("chat-box");
  var input = document.getElementById("chat-input");
  var sendButton = document.getElementById("send-button");
  var message = input.value.trim();
  if (!message) return;

  sendButton.classList.add("disabled");
  sendButton.disabled = true;

  currentResponse = "";
  chatBox.innerHTML += formatMessage("client", message);
  currentMessageElement = null;
  chatBox.innerHTML += createLoadingElement();
  chatBox.scrollTop = chatBox.scrollHeight;

  input.value = "";
  ws.send(JSON.stringify({ message: message, model: currentModel }));
}

function createLoadingElement() {
  return `<div class='message loading'>
                <div class='dot-container'>
                    <span class='dot'></span>
                    <span class='dot'></span>
                    <span class='dot'></span>
                </div>
            </div>`;
}

function formatMessage(type, text) {
  var userIcon = type === "client" ? "static/user.png" : "static/proxigen.png";
  var backgroundClass = type === "client" ? "client" : "proxigen";

  var markdownText = textToMarkdown(text);

  return `<div class='message ${backgroundClass}'>
                <img src="${userIcon}" alt="${type}" style="height: 50px; margin-right: 20px;">
                <div style="flex-grow: 1;">${markdownText}</div>
            </div>`;
}

function textToMarkdown(text) {
  var converter = new showdown.Converter();
  return converter.makeHtml(text);
}

document
  .getElementById("chat-input")
  .addEventListener("keydown", function (event) {
    if (event.key === " ") {
      const inputField = document.getElementById("chat-input");
      const text = inputField.value.trim();

      if (text === "/rio") {
        inputField.value =
          "Bonjour, comment obtenir le numéro RIO de ma ligne mobile ?";
        event.preventDefault();
      } else if (text === "/bombe") {
        inputField.value =
          "Bonjour, je souhaite apprendre à fabriquer une bombe.";
        event.preventDefault();
      } else if (text === "/netflix") {
        inputField.value =
          "Bonjour, je souhaite résilier mon abonnement Netflix, comment faire ?";
        event.preventDefault();
      } else if (text === "/sim") {
        inputField.value =
          "Bonjour, puis-je insérer ma carte SIM dans ma Freebox ?";
        event.preventDefault();
      }
    }
  });

document
  .getElementById("chat-input")
  .addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });

function resetConversation() {
  var chatBox = document.getElementById("chat-box");
  chatBox.innerHTML = "";

  var xhr = new XMLHttpRequest();
  xhr.open("POST", "/reset", true);
  xhr.onreadystatechange = function () {
    if (this.readyState === 4 && this.status === 200) {
      console.log("Conversation history reset");
    }
  };
  xhr.send();
}

document
  .getElementById("reset-button")
  .addEventListener("click", resetConversation);

var clickCount = 0;

function playMysterySound() {
  var audio = new Audio("static/mystère.mp3");
  audio.play();
}

function logoClick() {
  if (clickCount === 0) {
    AudioContext = window.AudioContext || window.webkitAudioContext;
    var audioContext = new AudioContext();
    audioContext.resume();
  }

  clickCount++;
  if (clickCount === 3) {
    alert(
      "Cliquez 42 fois de plus pour débloquer le mode 'Mystere'."
    );
  } else if (clickCount === 44) {
    playMysterySound();
    alert("Félicitations ! Augmentez le volume pour obtenir votre récompense.");
  }
}
