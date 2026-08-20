# Capacidades por tipo de cuenta

Qué se puede hacer con el proxy según el tipo de cuenta detrás del token:
**anónima** (sin token), **free** (`chatgpt_plan_type: free`) y **plan/Go**
(`chatgptgoplan`). Medido en vivo con [`smoke_test.py`](smoke_test.py) y
[`compare_accounts.py`](compare_accounts.py).

## Matriz de endpoints

| Capacidad / Endpoint | 🕶️ Anónima | 🆓 Free | 💳 Plan (Go) |
|---|:---:|:---:|:---:|
| Chat (`POST /v1/chat/completions`) | ✅ | ✅ | ✅ |
| Web search (en el chat, `web_search: true`) | ✅ | ✅ | ✅ |
| Modelos (`GET /v1/models`), `GET /v1/session/me` | ✅ | ✅ | ✅ |
| Traducir (`POST /v1/translate`) | ✅ | ✅ | ✅ |
| Límites (`GET /v1/limits`) | ✅ | ✅ | ✅ |
| Cuenta (`GET /v1/account`) | ❌ | ✅ | ✅ |
| Custom instructions (`GET·POST /v1/custom-instructions`) | ❌ | ✅ | ✅ |
| Suggestions (`GET /v1/suggestions`) | ❌ | ✅ | ✅ |
| Gizmos / chat como GPT (`GET /v1/gizmos`, `model:"g-..."`) | ❌ | ✅ (0 propios) | ✅ (tus GPTs) |
| Historial (`GET /v1/conversations`, `/{id}`) | ❌ | ✅ | ✅ |
| Biblioteca (`GET /v1/library`, `/usage`, `/{id}/download`, delete…) | ❌ | ✅ | ✅ |
| TTS (`POST /v1/audio/speech`, `GET /v1/audio/from-message`) | ❌ | ✅ | ✅ |
| STT (`POST /v1/audio/transcriptions`) | ❌ | ✅ | ✅ |
| **Imágenes** (`POST /v1/images/generations`) | ❌ | **❌ bloqueado** | ✅ |
| **Visión** (imagen como input en el chat, `image_url`) | ❌ | ✅ ² | ✅ |
| **Endpoints accesibles** | **5/15** | **16/17** | **17/17** |

`❌` en anónima = 401 "needs an authenticated account" (el endpoint requiere
cuenta; `synthesize`, `library`, `gizmos`, etc. no tienen variante
`/backend-anon`). Chat, translate, models, session/me y limits sí andan anónimos.

² **Visión** (mandar una imagen para que el modelo la analice) es distinto de
**generar** imágenes: la visión **no** está bloqueada en free (la generación sí).
Sube la imagen al file store de la cuenta (`POST /files` → PUT al blob →
`POST /files/{id}/uploaded`) y la adjunta al turno como `image_asset_pointer`.
Requiere cuenta (por eso `❌` en anónima). Verificado en vivo con go; free usa el
mismo camino autenticado, con menos cupo de `file_upload` (5 vs 80).

## Límites (cupos por período)

| Límite | 🕶️ Anónima | 🆓 Free | 💳 Go |
|---|:---:|:---:|:---:|
| `file_upload` | 3 | 5 | **80** |
| `paste_text_to_file` | 3 | 3 | **80** |
| `image_gen` | — | 5 *(no funciona, ver abajo)* | **106** |
| `reason` (modelos de razonamiento) | — | 8 | **300** |
| `deep_research` | — | 5 | 5 |
| `dictation` | 1 | — | — |
| **storage** | — | 540 MB (`limit_tier: free`) | **4.29 GB** (`limit_tier: go`) |

`model_limits: []` en las tres cuando no estás cerca del cap de mensajes; se
llena al acercarte (en free/anon llega mucho antes).

## Modelos

- **Comunes**: `gpt-5-5`, `gpt-5-6`, `gpt-5-3-mini`, `gpt-5-5-mini`, `gpt-5-6-mini`,
  `gpt-5-4-t-mini`, `gpt-5-6-t-mini`, `research`.
- **Free** agrega `auto` (router → modelos chicos) y `gpt-5-6-t-mini-mini` (variante
  aún más chica). Su `default_model` es `auto`.
- **Go** arranca en `gpt-5-6` por defecto.

## Por qué las imágenes fallan en free

No es un bug del proxy ni algo transitorio: es un **bloqueo de plan silencioso**.
El modelo **sí invoca** la herramienta de imágenes (`recipient: t2uay3k.sj1i4kz` =
DALL·E) — no la rechaza ni pide upgrade —, pero la generación **devuelve vacío**
(sin `image_asset_pointer`). El proxy ve 0 imágenes y responde
`503 "No image was generated"`. En Go la imagen se produce normalmente.

## Resumen

- **Anónima** → solo chat, traducir, modelos y límites. Nada de cuenta, historial,
  archivos, voz ni imágenes.
- **Free** → casi todo (16/17), **incluida voz completa (TTS + STT) y visión**. Lo
  único bloqueado es **generar imágenes** (la visión/input sí anda); los cupos son
  mínimos y el storage 8× menor.
- **Go** → todo, con cupos ~15–37× mayores, 8× más storage e imágenes.

La diferencia real **free vs go** no es la *superficie de API* (casi idéntica) sino
**imágenes + cupos + storage**.

## Herramientas de diagnóstico

```bash
python compare_accounts.py "<TOKEN>"                 # perfil de una cuenta
python compare_accounts.py "<TOKEN_A>" "<TOKEN_B>"   # diff entre dos

# smoke test (con el proxy corriendo con esa cuenta):
CHATGPT_ACCESS_TOKEN=<token> python -m uvicorn main:app --port 8899 &
python smoke_test.py                # read-only
python smoke_test.py --spend        # incluye chat/TTS/imágenes (gasta cuota)
```
