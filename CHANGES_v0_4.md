# PhysioSentinel Gait v0.4 Online

Cambios respecto a v0.3.1 local:

- Eliminadas rutas absolutas `D:\\Pose2Sim`.
- Uso de carpetas temporales compatibles con Linux/Streamlit Cloud.
- Configuración mínima de Pose2Sim generada en tiempo de ejecución; Pose2Sim completa sus parámetros por defecto.
- Eliminada la base SQLite persistente en la versión online para evitar asumir persistencia clínica en Community Cloud.
- Sesiones con identificador aleatorio para evitar colisiones.
- Se mantienen Pose2Sim + RTMPose / HALPE26, resultados 2D, ángulos dinámicos, vídeo anotado y conversión H.264.
- Se mantiene el modo de dos cámaras como base experimental, sin afirmar biomecánica 3D.
- Añadidos `requirements.txt`, `packages.txt`, `.gitignore` y configuración de Streamlit para despliegue.
