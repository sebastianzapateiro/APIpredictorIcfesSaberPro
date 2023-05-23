from flask import Flask, request, jsonify
import joblib
import pandas as pd
from IA import generar_respuesta
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Carga el modelo y el diccionario de mapeo
rf = joblib.load('modelopruebaforest8.joblib')
mapping_dict = joblib.load('mapping_dictGripPrueba71.joblib')

@app.route('/')
def index():
    return 'Welcome To the API'

@app.route('/predict', methods=['POST'])
def predict():
    # Obtiene los datos de entrada del JSON enviado en la solicitud
    input_data = request.get_json()

    # Mapea los valores categóricos a valores numéricos utilizando el diccionario de mapeo
    input_codes = {}
    for key, value in input_data.items():
        input_codes[key] = [mapping_dict[key][value]]

    # Crea un DataFrame con los datos de entrada codificados
    input_df = pd.DataFrame(input_codes)

    # Realiza la predicción con el modelo
    prediction = rf.predict(input_df)

    response = jsonify({'prediction': int(prediction[0])})
    response.headers.add("Access-Control-Allow-Origin", "*")

  
    return response





@app.route('/predict_tips', methods=['POST'])
def predict_tips():
    # Obtiene los datos de entrada del JSON enviado en la solicitud
    input_data = request.get_json()

    # Mapea los valores categóricos a valores numéricos utilizando el diccionario de mapeo
    input_codes = {}
    for key, value in input_data.items():
        input_codes[key] = [mapping_dict[key][value]]

    # Crea un DataFrame con los datos de entrada codificados
    input_df = pd.DataFrame(input_codes)

    # Realiza la predicción con el modelo
    prediction = rf.predict(input_df)

    # LlamaDA A la funcion generar_respuesta que se encuentra en el archivo IA :)
    #respuesta_openai = generar_respuesta(int(prediction[0]))

    response = jsonify({'prediction': int(prediction[0]), 'respuesta_Api_GPT': 'Para asegurar un buen puntaje en las pruebas ICFES, te recomiendo que tomes en consideración los siguientes consejos: \n\n1. Establece un horario de estudio y síguelo. Dedica cierto tiempo cada día a estudiar para que logres una adecuada preparación.\n\n2. Entiende los temas. Si no comprendes algo, investiga y pregunta al profesor para que te pueda ayudar.\n\n3. Practica siempre. Haz simulaciones de exámenes, resuelve ejercicios y práctica tus habilidades. \n\n4. Utiliza materiales de estudio adecuados. Busca libros que te expliquen el contenido de la manera más clara y sencilla para que lo entiendas fácilmente. \n\n5. Descansa. Estudiar mucho no siempre significa que obtendrás mejores resultados. Asegúrate de descansar para que tu mente se mantenga clara.\n\n6. Estudia en grupo. Estudiar con amigos puede ser una forma mucho más eficaz de adquirir conocimientos ya que podrás trabajar juntos para comprender mejor el contenido.'})
    response.headers.add("Access-Control-Allow-Origin", "*")
    #  predicción y la respuesta de GPT :V 
    return response
    
   


if __name__ == '__main__':
    app.run()