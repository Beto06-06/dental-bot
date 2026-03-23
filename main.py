import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from clinic_data import CLINIC_INFO, APPOINTMENT_KEYWORDS

load_dotenv()

app = FastAPI(title="Dental Bot API")

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
- Si preguntan por precios y no están definidos, indica que deben confirmarse por valoración o con recepción.
- Si el usuario quiere cita, indícale que debe escribir a recepción por WhatsApp o llamar a la clínica.
- SOLO puedes dar precios si están en la lista de precios.
- Si preguntan por un precio que no está, indica que debe confirmarse con la clínica.
- Responde en el mismo idioma en el que el usuario escribe.
- Si la pregunta del usuario coincide con una pregunta frecuente, utiliza exactamente esa respuesta.
- No reformules innecesariamente las respuestas del FAQ.

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
    }

    for item in CLINIC_INFO.get("faq", []):
        question = item["question"]
        keywords = faq_keywords.get(question, [])
        if any(keyword in text for keyword in keywords):
            return ChatResponse(
                reply=item["answer"],
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

    price_keywords = {
        "extracción compleja": [
            "extraccion compleja",
            "extracción compleja",
            "cordales",
            "cirugia",
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

    if any(keyword in text for keyword in price_intent_keywords):
        for service, keywords in price_keywords.items():
            if any(keyword in text for keyword in keywords):
                price = CLINIC_INFO.get("prices", {}).get(service)
                if price:
                    return ChatResponse(
                        reply=(
                            f"El precio de {service} es {price}. "
                            f"El costo puede variar según el diagnóstico. "
                            f"Te recomendamos agendar una valoración."
                        ),
                        redirect_to_human=False,
                    )

    ai_reply = ask_ai(message)

    return ChatResponse(
        reply=ai_reply,
        redirect_to_human=False,
    )