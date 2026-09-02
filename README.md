# PhysioSentinel Gait · Online v0.4

Versión de prueba para desplegar PhysioSentinel Gait en Streamlit Community Cloud.

## Qué hace

- Carga de vídeo frontal/posterior o lateral.
- Modo experimental de dos cámaras.
- Pose2Sim + RTMPose / HALPE26.
- Métricas temporales experimentales.
- Ángulos 2D proyectados de cadera, rodilla, tobillo y hombro.
- Vídeo con esqueleto y ángulos dinámicos.
- Conversión a MP4 H.264 compatible con navegador.

## Importante

Esta versión **no guarda historia clínica de forma persistente**. Los vídeos y resultados se almacenan en una carpeta temporal del servidor y pueden desaparecer cuando Streamlit reinicie la app. No introducir datos identificativos de pacientes durante esta fase de prueba.

Los ángulos mostrados son **proyecciones 2D**, no cinemática anatómica 3D. Cadencia, regularidad y asimetría siguen etiquetadas como experimentales hasta validar los eventos de contacto del pie.

## Despliegue en Streamlit Community Cloud

1. Crea un repositorio nuevo en GitHub, por ejemplo `physiosentinel-gait`.
2. Sube **todo el contenido de esta carpeta** a la raíz del repositorio.
3. En Streamlit Community Cloud crea una app nueva y selecciona ese repositorio.
4. Como archivo principal selecciona `streamlit_app.py`.
5. En **Advanced settings**, selecciona **Python 3.11**.
6. Pulsa **Deploy**.

La primera instalación puede tardar porque Pose2Sim/OpenSim y sus dependencias son grandes. La primera ejecución del análisis también puede tardar más si RTMPose necesita descargar su modelo.

## Si el despliegue falla

El objetivo de esta v0.4 es precisamente comprobar si Community Cloud dispone de recursos suficientes para Pose2Sim/RTMPose. Si el error es de memoria, tiempo de ejecución o límites del servidor, la interfaz Streamlit seguirá siendo reutilizable y el motor de análisis se podrá mover posteriormente a otro servidor.

## Archivos

- `streamlit_app.py`: aplicación principal.
- `requirements.txt`: dependencias Python.
- `packages.txt`: dependencias Linux.
- `.streamlit/config.toml`: configuración de Streamlit.
- `.gitignore`: evita subir vídeos, bases locales y secretos por accidente.
