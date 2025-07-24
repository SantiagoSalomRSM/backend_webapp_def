import os
import logging
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field # Pydantic V2 style
from typing import List, Optional, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
from openai import AzureOpenAI

# Setup logger and Azure Monitor:
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

# --- App FastAPI ---
app = FastAPI()

# --- Elegir el modelo a usar ---
MODEL = "gemini" 
# MODEL = "openai" 

if MODEL == "gemini":
    logger.info("Usando modelo Gemini para la generación de contenido.")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        logger.error("Error: La variable de entorno GEMINI_API_KEY no está configurada.")
    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            logger.info("Cliente de Gemini configurado correctamente.")
        except Exception as e:
            logger.error(f"Error configurando el cliente de Gemini: {e}")
    GEMINI_MODEL_NAME = "gemini-2.0-flash" # Use a valid model
elif MODEL == "openai":
    logger.info("Usando modelo Azure OpenAI para la generación de contenido.")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        logger.error("Error: Las variables de entorno AZURE_OPENAI_API_KEY o AZURE_OPENAI_ENDPOINT no están configuradas.")
    else:
        try:
            client = AzureOpenAI(
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                api_version="2025-02-01-preview"
            )
        except Exception as e:
            logger.error(f"Error configurando el cliente de OpenAI: {e}")


# --- Constantes de Estado y TTL ---
STATUS_PROCESSING = "processing"
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
STATUS_NOT_FOUND = "not_found" # Estado implícito si no existe la key
GEMINI_ERROR_MARKER = "ERROR_PROCESSING_GEMINI" # Marcador en el resultado
DEEPSEEK_ERROR_MARKER = "ERROR_PROCESSING_DEEPSEEK" # Marcador en el resultado
OPENAI_ERROR_MARKER = "ERROR_PROCESSING_OPENAI" # Marcador en el resultado

# --- Modelos Pydantic ---
class TallyOption(BaseModel):
    id: str
    text: str
    
class TallyField(BaseModel):
    key: str
    label: Optional[str]
    value: Any
    type: str
    options: Optional[List[TallyOption]] = None

class TallyResponseData(BaseModel):
    responseId: str
    submissionId: str
    formName: str
    fields: List[TallyField]

class TallyWebhookPayload(BaseModel):
    eventId: str
    eventType: str
    data: TallyResponseData

# Placeholder model for PUT data (adjust as needed)
class UpdateResultPayload(BaseModel):
    new_result: str
    reason: Optional[str] = None

# Inicializar el cliente de OpenAI

def summarize_payload(payload: TallyWebhookPayload) -> str:
    """Genera un resumen entendible del Tally payload."""
    lines = ["Respuestas:"]
    for field in payload.data.fields:
        label = field.label or field.key
        value = field.value
        # Si el valor es una lista y tiene opciones, mapeamos los IDs a texto
        if isinstance(value, list) and field.options:
            id_to_text = {opt.id: opt.text for opt in field.options}
            value_texts = [id_to_text.get(v, v) for v in value]
            value_str = ", ".join(value_texts)
        else:
            value_str = str(value)
        lines.append(f"- {label}: {value_str}")
    return "\n".join(lines)

def detect_form_type(payload: TallyWebhookPayload) -> str:
    """Detecta el form type basándose en la primera label o key."""
    mode = "unknown"  # Valor por defecto
    if payload.data.formName:
        formName = payload.data.formName
        logger.info(f"Form Name: {formName}")
        logger.info(f"Form Name stripped: {formName.strip()}")
        return formName.strip()
    return mode

def load_prompt_from_file(prompt_name: str) -> str:
    """Carga un prompt desde un archivo en la carpeta de prompts."""
    path = f"prompts/{prompt_name}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: Prompt file '{path}' not found."
    
def generate_prompt(payload: TallyWebhookPayload, submission_id: str, mode: str) -> str:
    """Genera un prompt basado en el tipo de formulario."""
  
    logger.info(f"[{submission_id}] Generando Prompt.")

    if mode == "unknown":
        logger.warning(f"[{submission_id}] Tipo de formulario desconocido. Usando prompt genérico.")
        prompt_parts = ["Analiza la siguiente respuesta de un formulario\n", "Proporciona un resumen o conclusión en formato markdown:\n\n"]
    else:
        # Extraer el prompt de la carpeta de prompts
        prompt_text = load_prompt_from_file(mode)
        prompt_parts = [prompt_text]

        # ... ( lógica para construir el prompt con payload.data.fields) ... 
        for field in payload.data.fields:
            label = field.label
            label_str = "null" if label is None else str(label).strip()
            value = field.value
            value_str = ""
            if isinstance(value, list):
                try:
                    value_str = f'"{",".join(map(str, value))}"'
                except Exception as e:
                    logger.error(f"[{submission_id}] Error convirtiendo lista a string: {e}")
                    value_str = "[Error procesando lista]"
            elif value is None:
                value_str = "null"
            else:
                value_str = str(value)
            prompt_parts.append(f"Pregunta: {label_str} - Respuesta: {value_str}")

# -------------------------------------------------
    full_prompt = "".join(prompt_parts)
    return full_prompt

# --- Lógica para interactuar con OpenAI ---
async def generate_openai_response(submission_id: str, prompt: str, mode: str, cur: Any, conn: Any) -> None:
    """Genera una respuesta de OpenAI y actualiza la base de datos con el resultado."""
    logger.info(f"[{submission_id}] Iniciando tarea OpenAI.")
    
    try:
        # --- Llamada a OpenAI API ---
        response = client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),  
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,  # algo razonable
                temperature=0.7
            )

        result_text = response.choices[0].message.content

        # --- Limpieza del mensaje ---
        clean_text = result_text.strip()

        if clean_text.startswith("```html"):
            clean_text = clean_text.removeprefix("```html")

        if clean_text.endswith("```"):
            clean_text = clean_text.removesuffix("```")

        result_text = clean_text.strip()

        # --- Actualizar Base de datos con el resultado ---
        if result_text:
            if mode == "CONSULTING":
                try:
                    cur.execute("""UPDATE formai_db 
                                SET status = %s, result_consulting = %s 
                                WHERE submission_id = %s""", 
                                (STATUS_SUCCESS, result_text, submission_id))
                    conn.commit()
                    logger.info(f"[{submission_id}] Resultado guardado en la base de datos.")
                except Exception as e:
                    logger.error(f"[{submission_id}] Error guardando resultado en base de datos: {e}")
            else:
                try:
                    cur.execute("""UPDATE formai_db 
                                SET status = %s, result_client = %s 
                                WHERE submission_id = %s""", 
                                (STATUS_SUCCESS, result_text, submission_id))
                    conn.commit()
                    logger.info(f"[{submission_id}] Resultado guardado en la base de datos.")
                except Exception as e:
                    logger.error(f"[{submission_id}] Error guardando resultado en base de datos: {e}")
        else:
            # Si no hay texto válido, guardar error
            try:
                cur.execute("""UPDATE formai_db 
                            SET status = %s, result_client = %s, result_consulting = %s 
                            WHERE submission_id = %s""", 
                            (STATUS_ERROR, OPENAI_ERROR_MARKER, OPENAI_ERROR_MARKER, submission_id))
                conn.commit()
                logger.warning(f"[{submission_id}] Resultado vacío. Marcador de error guardado en la base de datos.")
            except Exception as e:
                logger.error(f"[{submission_id}] Error guardando marcador de error en base de datos: {e}")

    except Exception as e:
        logger.error(f"[{submission_id}] Error llamando a OpenAI: {e}")
        try:
            # Intenta guardar el estado de error incluso si OpenAI falló
            cur.execute("""UPDATE formai_db 
                        SET status = %s, result_client = %s, result_consulting = %s 
                        WHERE submission_id = %s""", 
                        (STATUS_ERROR, OPENAI_ERROR_MARKER, OPENAI_ERROR_MARKER, submission_id))
            conn.commit()
            logger.error(f"[{submission_id}] Estado de error guardado en la base de datos.")
        except Exception as e:
            logger.error(f"[{submission_id}] Error guardando estado de error en base de datos: {e}")
    
    logger.info(f"[{submission_id}] Tarea OpenAI finalizada.")

async def generate_gemini_response(submission_id: str, prompt: str, mode: str, cur: Any, conn: Any) -> None:
    """Genera una respuesta de Gemini y actualiza base de datos con el resultado."""
    logger.info(f"[{submission_id}] Iniciando tarea Gemini.")
    
    try:
        # --- Llamada a Gemini API (lógica sin cambios) ---
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = await model.generate_content_async(prompt)
        result_text = None # Inicializa result_text

        if response and hasattr(response, 'text') and response.text:
             result_text = response.text
             logger.info(f"[{submission_id}] Respuesta de Gemini recibida (text)")
        elif response and hasattr(response, 'parts'):
             result_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
             if result_text:
                logger.info(f"[{submission_id}] Respuesta de Gemini recibida (desde parts).")
             else:
                 logger.warning(f"[{submission_id}] Respuesta de Gemini (parts) sin texto")
                 result_text = None # Asegura que no se guarde si está vacío
        else:
            logger.warning(f"[{submission_id}] Respuesta Gemini inesperada o sin texto")
 
        # --- Actualizar base de datos con el resultado ---
        if result_text:
            if mode == "CONSULTING":
                try:
                    cur.execute("""UPDATE formai_db 
                                SET status = %s, result_consulting = %s 
                                WHERE submission_id = %s""", 
                                (STATUS_SUCCESS, result_text, submission_id))
                    conn.commit()
                    logger.info(f"[{submission_id}] Resultado guardado en la base de datos.")
                except Exception as e:
                    logger.error(f"[{submission_id}] Error guardando resultado en base de datos: {e}")
            else:
                try:
                    cur.execute("""UPDATE formai_db 
                                SET status = %s, result_client = %s 
                                WHERE submission_id = %s""", 
                                (STATUS_SUCCESS, result_text, submission_id))
                    conn.commit()
                    logger.info(f"[{submission_id}] Resultado guardado en la base de datos.")
                except Exception as e:
                    logger.error(f"[{submission_id}] Error guardando resultado en base de datos: {e}")
        else:
            # Si no hay texto válido, guardar error
            try:
                cur.execute("""UPDATE formai_db 
                            SET status = %s, result_client = %s, result_consulting = %s 
                            WHERE submission_id = %s""", 
                            (STATUS_ERROR, OPENAI_ERROR_MARKER, OPENAI_ERROR_MARKER, submission_id))
                conn.commit()
                logger.warning(f"[{submission_id}] Resultado vacío. Marcador de error guardado en la base de datos.")
            except Exception as e:
                logger.error(f"[{submission_id}] Error guardando marcador de error en base de datos: {e}")

    except Exception as e:
        logger.error(f"[{submission_id}] Error llamando a OpenAI: {e}")
        try:
            # Intenta guardar el estado de error incluso si OpenAI falló
            cur.execute("""UPDATE formai_db 
                        SET status = %s, result_client = %s, result_consulting = %s 
                        WHERE submission_id = %s""", 
                        (STATUS_ERROR, OPENAI_ERROR_MARKER, OPENAI_ERROR_MARKER, submission_id))
            conn.commit()
            logger.error(f"[{submission_id}] Estado de error guardado en la base de datos.")
        except Exception as e:
            logger.error(f"[{submission_id}] Error guardando estado de error en base de datos: {e}")
    
    logger.info(f"[{submission_id}] Tarea OpenAI finalizada.")

# --- Endpoints FastAPI ---
@app.post("/webhook")
async def handle_tally_webhook(payload: TallyWebhookPayload):

    # --- Configuración base de datos ---
    try:
        host = os.getenv("POSTGRES_HOST")
        dbname = os.getenv("POSTGRES_DB")
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        port = int(os.getenv("POSTGRES_PORT", 5432))  # Usa 5432 como valor por defecto si no se encuentra
    except KeyError as e:
        logger.error(f"Missing environment variable: {e}")
        return {"error": f"Missing environment variable: {e}"}, 500 

    try:
        # Conectar a la base de datos PostgreSQL con psycopg2   
        conn = psycopg2.connect(
                host=host,
                database=dbname,
                user=user,
                password=password,
                port=port,
                sslmode="require"
            )
        cur = conn.cursor()
    except Exception as e:
        logger.error(f"Error connecting to the database: {e}")
        return {"error": f"Error connecting to the database: {e}"}, 500
    
    # --- Lectura de datos ---
    logger.info(f"payload recibido: {payload}")
    submission_id = payload.data.submissionId
    logger.info(f"[{submission_id}] Webhook recibido. Verificando (ID: {submission_id}).")
    logger.info(f"[{submission_id}] Event ID: {payload.eventId}, Event Type: {payload.eventType}")

    try:
        # Verificar si ya existe estado en la tabla
        cur.execute("SELECT status FROM formai_db WHERE submission_id = %s", (submission_id,))
        row = cur.fetchone()
        if row is None:
            logger.info(f"[{submission_id}] No se encontró estado previo. Creando nuevo registro.")
        else:
            logger.info(f"[{submission_id}] Estado previo encontrado: {row[0]}.")
            status = cur.fetchone()[0]
            if status:
                if status == STATUS_PROCESSING:
                    logger.warning(f"[{submission_id}] Webhook ignorado: ya está en estado '{STATUS_PROCESSING}'.")
                    return {"message": f"Webhook ignored: already in processing state '{STATUS_PROCESSING}'."}, 200
                else:
                    logger.warning(f"[{submission_id}] Webhook ignorado: ya tiene estado final {status}.")
                    return {"message": f"Webhook ignored: already has final state '{status}'."}, 200
        
        # Extraer información relevante del formulario
        form_type = detect_form_type(payload)
        response = summarize_payload(payload)
        
        cur.execute("""INSERT INTO formai_db (submission_id, status, result_client, result_consulting, user_responses, form_type, created_at, updated_at) 
                    VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())""", 
                    (submission_id, STATUS_PROCESSING, None, None, response, form_type))
        conn.commit()

        # Si llegamos aquí, la key se creó y se puso en 'processing'
        logger.info(f"[{submission_id}] Estado '{STATUS_PROCESSING}' establecido en BD.")

# -------------------------------------------------
        if MODEL == "openai":
            # --- Generación del Prompt modularizada ---
            prompt_cliente = generate_prompt(payload, submission_id, form_type)
            logger.debug(f"[{submission_id}] Prompt para Gemini: {prompt_cliente[:200]}...")
        
            # --- Iniciar Tarea en Segundo Plano ---
            await generate_openai_response(submission_id, prompt_cliente, form_type, cur, conn)
            logger.info(f"[{submission_id}] Tarea de Gemini iniciada en segundo plano.")
        
            # --- Generación del Prompt para consultoría ---
            prompt_consulting = generate_prompt(payload, submission_id, "CONSULTING")
            logger.debug(f"[{submission_id}] Prompt para Gemini (Consulting): {prompt_consulting[:200]}...")

            # --- Iniciar Tarea en Segundo Plano (después de respuesta cliente) ---
            await generate_openai_response(submission_id, prompt_consulting, "CONSULTING", cur, conn)
            logger.info(f"[{submission_id}] Tarea de Gemini iniciada en segundo plano.")

        elif MODEL == "gemini":
            # --- Generación del Prompt modularizada ---
            prompt_cliente = generate_prompt(payload, submission_id, form_type)
            logger.debug(f"[{submission_id}] Prompt para Gemini: {prompt_cliente[:200]}...")
        
            # --- Iniciar Tarea en Segundo Plano ---
            await generate_gemini_response(submission_id, prompt_cliente, form_type, cur, conn)
            logger.info(f"[{submission_id}] Tarea de Gemini iniciada en segundo plano.")
        
            # --- Generación del Prompt para consultoría ---
            prompt_consulting = generate_prompt(payload, submission_id, "CONSULTING")
            logger.debug(f"[{submission_id}] Prompt para Gemini (Consulting): {prompt_consulting[:200]}...")

            # --- Iniciar Tarea en Segundo Plano (después de respuesta cliente) ---
            await generate_gemini_response(submission_id, prompt_consulting, "CONSULTING", cur, conn)
            logger.info(f"[{submission_id}] Tarea de Gemini iniciada en segundo plano.")

        return {"message": f"Webhook processed successfully for submission {submission_id}."}, 200
    
    except Exception as e:
        logger.error(f"[{submission_id}] Error procesando webhook: {e}", exc_info=True)
        # Devolver error 500 si algo falla aquí es crítico
        return {"error": f"Error processing webhook: {e}"}, 500
    finally:
        # Cerrar conexión a la base de datos
        if cur:
            cur.close()
        if conn:
            conn.close()
        logger.info(f"[{submission_id}] Conexión a BD cerrada.")

@app.get("/")
async def root():
    """Endpoint raíz simple para verificar que la app funciona."""
    return {"message": "Hola! Soy el procesador de Tally a Gemini."}