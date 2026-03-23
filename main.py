import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

from clinic_data import CLINIC_INFO, APPOINTMENT_KEYWORDS

load_dotenv()

app = FastAPI(title="Dental Bot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    redirect_to_human: bool
    whatsapp_link: Optional[str] = None


def normalize_text(text: str) -> str:
    return text.strip().lower()


def wants_appointment(message: str) -> bool:
    text = normalize_text(message)
    return any(keyword in text for keyword in APPOINTMENT_KEYWORDS)


def build_whatsapp_link() -> str:
    phone = CLINIC_INFO["whatsapp"]
    text = "Hola, me gustaría agendar una cita."
    return f"https://wa.me/{phone}?text={text.replace(' ', '%20')}"


def format_doctors() -> str:
    doctors = CLINIC_INFO.get("doctors", [])
    if not doctors:
        return "No hay información disponible sobre los doctores."
    return ", ".join(
        [f"{doctor['name']} ({doctor['specialty']})" for doctor in doctors]
    )


def get_system_prompt() -> str:
    return f"""
Eres un asistente virtual de la clínica {CLINIC_INFO['name']}.

Tu trabajo es:
1. Responder preguntas frecuentes de forma clara y profesional.
2. Usar únicamente la información proporcionada.
3. No inventar precios, horarios, servicios ni información médica.
4. No diagnosticar ni recomendar tratamientos definitivos.
5. Si el usuario quiere agendar una cita, NO agendes tú. Debes indicarle que la coordinación la hace la recepcionista.
6. Si no sabes algo, di que debe confirmarlo directamente con la clínica.

Información de la clínica:
- Nombre: {CLINIC_INFO['name']}
- Ubicación: {CLINIC_INFO['location']}
- Horario: {CLINIC_INFO['hours']}
- Teléfono: {CLINIC_INFO['phone']}
- WhatsApp: {CLINIC_INFO['whatsapp']}
- Correo: {CLINIC_INFO['email']}
- Servicios: {', '.join(CLINIC_INFO['services'])}
- Métodos de pago: {', '.join(CLINIC_INFO['payment_methods'])}
- Idiomas: {', '.join(CLINIC_INFO.get('languages', []))}
- Doctores: {format_doctors()}

Preguntas frecuentes:
{chr(10).join([f"- {item['question']}: {item['answer']}" for item in CLINIC_INFO['faq']])}

Reglas importantes:
- Mantén respuestas breves.
- Nunca des una lista completa de precios si el usuario no menciona servicios específicos.
- Si preguntan por precios y no están definidos, indica que deben confirmarse por valoración o con recepción.
- Si el usuario quiere cita, indícale que debe escribir a recepción por WhatsApp o llamar a la clínica.
- SOLO puedes dar precios si están en la lista de precios.
- Si preguntan por un precio que no está, indica que debe confirmarse con la clínica.
- Responde en el mismo idioma en el que el usuario escribe.
- Si la pregunta del usuario coincide con una pregunta frecuente, utiliza exactamente esa respuesta.
- No reformules innecesariamente las respuestas del FAQ.
- No des una lista completa de precios a menos que el usuario pida varios servicios específicos.
- Si el usuario pregunta por precios en general sin mencionar un servicio, pídele que indique cuál tratamiento le interesa.
- Si el usuario menciona un solo servicio, responde solo con el precio de ese servicio.
- Si el usuario menciona varios servicios, responde solo con los precios de esos servicios.

Precios:
{chr(10).join([f"  - {service}: {price}" for service, price in CLINIC_INFO.get('prices', {}).items()])}
"""


def ask_ai(user_message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": user_message},
        ],
        temperature=0.4,
    )

    return response.choices[0].message.content.strip()


@app.get("/")
def root():
    return {"message": "Dental Bot API funcionando"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        return ChatResponse(
            reply="Por favor escribe tu consulta.",
            redirect_to_human=False,
        )

    if wants_appointment(message):
        whatsapp_link = build_whatsapp_link()
        return ChatResponse(
            reply=(
                f"Con gusto te ayudamos. Para coordinar tu cita, puedes escribirnos por WhatsApp aquí: "
                f"{whatsapp_link} o llamar al {CLINIC_INFO['phone']} dentro de nuestro horario de atención."
            ),
            redirect_to_human=True,
            whatsapp_link=whatsapp_link,
        )

    if "ingles" in message.lower() or "inglés" in message.lower():
        return ChatResponse(
            reply="Sí, contamos con atención en inglés. Nuestro equipo puede ayudarte sin problema.",
            redirect_to_human=False,
        )

    text = message.lower()

    faq_keywords = {
        "¿Cómo puedo llegar a la clínica?": [
            "como llegar",
            "cómo llegar",
            "waze",
            "google maps",
        ],
        "¿Están cerca de algún punto de referencia importante?": [
            "hotel la siesta",
            "punto de referencia",
            "cerca de",
        ],
        "¿Atienden con cita o también sin cita?": [
            "sin cita",
            "con cita",
            "previa cita",
        ],
        "¿Qué pasa si llego tarde a mi cita?": [
            "llego tarde",
            "llegar tarde",
        ],
        "¿Atienden fines de semana?": [
            "sabado",
            "sábado",
            "fin de semana",
        ],
        "¿Atienden en días feriados?": [
            "feriado",
            "feriados",
        ],
        "¿Cuál es la última hora para agendar cita?": [
            "ultima hora",
            "última hora",
            "6 de la tarde",
        ],
        "¿Atienden adultos mayores?": [
            "adultos mayores",
        ],
        "¿Atienden emergencias dentales?": [
            "emergencia",
            "urgencia",
        ],
        "¿Realizan radiografías en la clínica?": [
            "radiografia",
            "radiografía",
            "rayos x",
        ],
        "¿La limpieza dental duele?": [
            "limpieza duele",
        ],
        "¿Cuánto dura una limpieza dental?": [
            "cuanto dura limpieza",
            "cuánto dura limpieza",
            "horario limpieza",
        ],
        "¿El blanqueamiento dental es seguro?": [
            "blanqueamiento seguro",
        ],
        "¿Cuánto dura el efecto del blanqueamiento?": [
            "cuanto dura blanqueamiento",
            "cuánto dura blanqueamiento",
        ],
        "¿Cuánto tiempo dura una endodoncia?": [
            "cuanto dura endodoncia",
            "cuánto dura endodoncia",
        ],
        "¿Las extracciones dentales son dolorosas?": [
            "extracciones duelen",
            "extraccion duele",
            "extracción duele",
            "sacar muela duele",
        ],
        "¿Atienden en inglés?": [
            "atienden en ingles",
            "atienden en inglés",
        ],
        "¿Cuál es el horario?": [
            "horario",
            "horario porfavor",
            "horario por favor",
            "a que hora atienden",
            "a qué hora atienden",
        ],
    }

    for item in CLINIC_INFO.get("faq", []):
        question = item["question"]
        keywords = faq_keywords.get(question, [])
        if any(keyword in text for keyword in keywords):
            return ChatResponse(
                reply=item["answer"],
                redirect_to_human=False,
            )

    price_keywords = {
        "extracción compleja": [
            "extraccion compleja",
            "extracción compleja"
        ],
        "extracción": [
            "extraccion",
            "extracción",
            "sacar muela",
        ],
        "limpieza dental": [
            "limpieza",
            "profilaxis",
        ],
        "resinas": [
            "resina",
            "resinas",
            "calza",
        ],
        "sellantes": [
            "sellante",
            "sellantes",
        ],
        "blanqueamiento dental": [
            "blanqueamiento",
            "blanquear dientes",
        ],
        "radiografía intraoral": [
            "radiografia",
            "radiografía",
            "rayos x",
        ],
    }

    if (
    ("cordales" in text or "cirugia de cordales" in text or "cirugía de cordales" in text)
    and any(keyword in text for keyword in [
        "precio", "cuesta", "cuánto cuesta", "cuanto cuesta", "vale", "costo", "price", "how much"
    ])):
        return ChatResponse(
        reply=(
            "El costo de la cirugía de cordales puede variar según la valoración del caso. "
            "Te recomendamos agendar una cita para brindarte información más precisa."
        ),
        redirect_to_human=False,
    )

    general_price_requests = [
        "precios",
        "precio de servicios",
        "lista de precios",
        "cuáles son los precios",
        "quiero saber precios",
        "what are your prices",
        "price list",
    ]

    if any(phrase in text for phrase in general_price_requests) and not any(
        kw in text for kws in price_keywords.values() for kw in kws
    ):
        return ChatResponse(
            reply=(
                "Con gusto. Podemos brindarte el precio del tratamiento que te interese. "
                "Por favor indícanos cuál servicio deseas consultar, por ejemplo: limpieza dental, extracción, blanqueamiento o radiografía intraoral."
            ),
            redirect_to_human=False,
        )

    price_intent_keywords = [
        "precio",
        "cuesta",
        "cuánto cuesta",
        "cuanto cuesta",
        "vale",
        "costo",
        "cost",
        "price",
        "how much",
    ]

    if any(keyword in text for keyword in price_intent_keywords):
        matched_prices = []

        for service, keywords in price_keywords.items():
            if any(keyword in text for keyword in keywords):
                price = CLINIC_INFO.get("prices", {}).get(service)
                if price:
                    matched_prices.append(f"{service}: {price}")

        if matched_prices:
            if len(matched_prices) == 1:
                reply_text = (
                    f"El precio de {matched_prices[0]}. "
                    f"El costo puede variar según el diagnóstico. "
                    f"Te recomendamos agendar una valoración."
                )
            else:
                reply_text = (
                    "Estos son los precios de los servicios consultados:\n- "
                    + "\n- ".join(matched_prices)
                    + "\nEl costo puede variar según el diagnóstico. "
                      "Te recomendamos agendar una valoración."
                )

            return ChatResponse(
                reply=reply_text,
                redirect_to_human=False,
            )

    ai_reply = ask_ai(message)

    return ChatResponse(
        reply=ai_reply,
        redirect_to_human=False,
    )
#LUNES 23 DE MARZO BOT FUNCIONANDO A LA 1:21pm 