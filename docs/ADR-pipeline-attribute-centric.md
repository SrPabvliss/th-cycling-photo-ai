# ADR · Modelo de datos atributo-céntrico para integración con backend

**Estado:** Aprobado | **Fecha:** 2026-05-06
**Autor:** Pablo Villacrés Morales
**Aplica a:** `pipeline/` (Cycling Photo AI service) — contrato API consumido por backend NestJS
**Relacionados:** ADR-013 (`scope_simplification` — filtro de `cyclist_with_bike`), ADR-019 §addendum (color Gemini-only)

---

## 1. Contexto

Durante la planificación de la integración pipeline ↔ backend NestJS surgió el problema de cómo agrupar items detectados por ciclista cuando una foto contiene múltiples personas (escenarios de pelotón / drafting habituales en ciclismo).

Inicialmente se propuso agregar un campo `cyclist_id: int | null` a `DetectionItem`, `BibReadingItem` y `ColorAnalysisItem` en la respuesta del endpoint `POST /pipeline`. La asignación se haría en el orchestrator vía heurística espacial (Intersection-over-Smaller con bicycle bbox como ancla, fallback a `null` cuando ambiguo).

Esta propuesta se evaluó contra el caso de uso real del producto Titan TV y se descartó por las razones documentadas en §3.

---

## 2. Decisión

**El pipeline NO emite `cyclist_id`.** La respuesta API mantiene listas planas (`detections`, `bib_readings`, `color_analyses`) con `bbox_source` como única información espacial por ítem.

**El backend modela los datos con esquema atributo-céntrico**, no entidad-céntrico:
- Cada foto guarda los items detectados como atributos del set "esta foto contiene".
- Las búsquedas operan sobre set membership: `WHERE photo.placas CONTAINS '20'`, `WHERE photo.colores_casco CONTAINS 'rojo'`.
- No existe entidad "ciclista" individualizable a nivel base de datos.

---

## 3. Razonamiento

### 3.1 El caso de uso es búsqueda por atributos, no por entidad

El producto Titan TV permite al usuario final buscar fotos por:
- Número de placa (búsqueda principal).
- Color de casco / ropa / bicicleta dentro de un evento (búsqueda secundaria).

Ninguno de estos casos requiere "identidad" de ciclista. Una consulta del tipo "fotos del ciclista 20" se resuelve correctamente con el modelo atributo-céntrico: cualquier foto cuyo set de placas detectadas contenga `20` matchea, independientemente de qué otra placa también aparezca.

### 3.2 El pipeline no tiene capacidad real de identidad cross-foto

Sin re-identificación visual (re-id), sin tracking entre fotos consecutivas, y sin features biométricas, el pipeline solo puede asignar un índice arbitrario `0..N-1` válido para una sola foto. Ese índice no es comparable entre fotos: el "ciclista 0" de la foto A no es necesariamente el "ciclista 0" de la foto B.

Exponer `cyclist_id` con un significado solo válido intra-foto es **pseudo-precisión**: un consumidor podría asumir incorrectamente que el ID es estable o significativo cross-foto, cuando solo es una etiqueta interna para una agrupación heurística.

### 3.3 La heurística espacial es frágil en escenas con drafting

La asignación por Intersection-over-Smaller con bicycle bbox como ancla funciona bien en fotos con un solo ciclista y aceptablemente en multi-ciclista bien separado. Falla silenciosamente en drafting (bicycles se solapan >50%), que es la situación más común en eventos ciclistas profesionales.

Mitigarlo con `cyclist_id: null` cuando ambiguo es honesto pero no resuelve el problema: el consumidor termina con datos parciales y debe implementar fallback igualmente.

### 3.4 El revisor humano cierra el loop visualmente

El frontend muestra la foto original con bboxes superpuestos por clase + crops extraídos. El revisor confirma o corrige cada item del set sin necesitar pre-agrupación. La asociación visual entre items y ciclistas queda implícita en el render, no en una columna FK de la base de datos.

### 3.5 El esfuerzo va donde tiene contexto

Repensar el esquema de la base de datos del backend (zona conocida) es menor riesgo y menor esfuerzo que introducir lógica espacial en el orchestrator (zona donde el pipeline no debería razonar sobre identidad).

---

## 4. Trade-offs aceptados explícitamente

### 4.1 Búsquedas compuestas atómicas pueden devolver falsos positivos

Una consulta del tipo "fotos donde el ciclista 20 lleva casco rojo" se traduce, bajo este modelo, en "fotos donde aparece la placa 20 Y aparece un casco rojo". Si la foto contiene placas `20 + 21` y cascos `rojo + azul`, matchea aunque la 20 lleve azul.

**Mitigación:** el frontend NO expone búsquedas compuestas atómicas atributo+atributo. Solo búsquedas atómicas por atributo.

Si en el futuro el producto requiere búsqueda compuesta, hay dos caminos:
- (a) Exponer la limitación explícitamente en el UI ("búsqueda por aparición conjunta, no por co-ocurrencia exacta").
- (b) Reabrir la decisión y agregar `cyclist_id` heurístico con todas sus limitaciones documentadas.

### 4.2 Analytics agregadas por ciclista no son posibles

Métricas del tipo "% de ciclistas con casco rojo en evento X" no se calculan: contaríamos cascos, no ciclistas. Si dos fotos del mismo ciclista aparecen, su casco se cuenta dos veces.

**Mitigación:** este caso no forma parte del MVP de Titan TV. Si en el futuro se requiere analytics agregada, requerirá identidad cross-foto (re-id), que está fuera del scope de la tesis.

### 4.3 UI no puede agrupar items por ciclista en cards

El revisor ve items sueltos (cada placa, cada color por región) en lugar de cards por ciclista. Ligero overhead UX en fotos con 3+ ciclistas, pero compensado por la simplicidad de implementación y la honestidad del modelo (no muestra agrupaciones que el sistema no puede garantizar).

---

## 5. Alternativas consideradas

### Opción A · `cyclist_id` heurístico con IoS
- Bicycle bbox como ancla, IoS ≥ 0.7 para asignar items, `null` cuando ambiguo.
- **Esfuerzo:** ~1.5 días pipeline + tests + integración.
- **Rechazada por:** pseudo-precisión + fragilidad en drafting + duplicación de lógica espacial entre pipeline y backend.

### Opción B · Re-introducir `cyclist_with_bike` como ancla interna
- Revertir parte de ADR-013 internamente, usar `cyclist_with_bike` para agrupar.
- **Rechazada por:** reabre decisión cerrada + sigue sin resolver drafting + el caso de uso no lo justifica.

### Opción C · Asignación tipo Hungarian
- Optimización global por minimización de costo (distancia centroides).
- **Rechazada por:** mayor complejidad sin beneficio claro vs heurística simple + sigue siendo enfoque entidad-céntrico.

### Opción D · No modelar identidad (esta decisión)
- Pipeline emite listas planas. Backend modela atributos de foto. UI muestra items sueltos para revisión.
- **Adoptada.**

---

## 6. Consecuencias

### Para el pipeline (este repo)
- `pipeline/schemas.py` se mantiene sin cambios respecto al estado post-commit `95f0636`.
- No se agregan campos `cyclist_id` ni similares.
- El contrato API queda **congelado en versión `schema_version="1.0"`**.

### Para el backend NestJS
- Esquema BD es atributo-céntrico (tablas planas a nivel foto, sin tabla `CyclistGroup`).
- Endpoints de búsqueda hacen JOIN simple por `photo_id` + filter por valor de atributo.
- Reviewer UI muestra items sueltos con bboxes superpuestos.
- Detalle del esquema recomendado en `docs/backend_integration_handoff.md` §6.

### Para la tesis
- Hallazgo metodológico publishable: rechazo deliberado de identidad heurística como pseudo-precisión cuando el caso de uso es búsqueda por atributos.
- Justifica la simplicidad arquitectónica del pipeline contra el reflejo natural de "agregar más estructura".

---

## 7. Hallazgo metodológico para la tesis

> "La integración pipeline ↔ backend reveló una tensión entre dos modelos mentales para los datos: entidad-céntrico (modelar ciclistas individuales) vs atributo-céntrico (modelar atributos de la foto). La decisión arquitectónica final fue rechazar el modelo entidad-céntrico porque (a) el caso de uso del producto es búsqueda por atributos, no consultas por identidad; (b) el pipeline carece de capacidad de re-identificación cross-foto que sostendría una noción estable de identidad; (c) introducir agrupación heurística intra-foto sería pseudo-precisión: un dato que parece significativo pero solo refleja una decisión geométrica frágil. Lecciones: la complejidad debe vivir donde el dominio la sostenga, y un sistema honesto sobre sus límites es más útil que uno que aparenta capacidades que no tiene."

---

**Aprobación:** Pablo Villacrés Morales | 2026-05-06
