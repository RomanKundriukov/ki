# Impots
```python
import tensorflow as tf
import numpy as np
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
```

# Initialisieren den Logger
```python
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
```

# Vorbereiten die Daten
```python
tabelle_pfad="./TrainingsData/C&F.xlsx"

df = pd.read_excel(tabelle_pfad)

celcius_q = np.array(df['C'], dtype=float)
fahrenheit_a = np.array(df['F'], dtype=float)

celcius_q_train = celcius_q[:300]
fahrenheit_a_train = fahrenheit_a[:300]

# Testdaten (Labels - ab Index 200)
celcius_q_label = celcius_q[300:]
fahrenheit_a_label = fahrenheit_a[300:]
```

# Erstellen ein Modell

```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),               # Eingabe-Schicht explizit definiert
    tf.keras.layers.Dense(16, activation='relu'),  # Verborgene Schicht
    tf.keras.layers.Dense(8, activation='relu'),   # Weitere verborgene Schicht
    tf.keras.layers.Dense(1)                       # Ausgabeschicht
])

# Modellübersicht anzeigen
model.summary()

# Modell kompilieren
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),  # Kleinere Lernrate
    loss='mean_squared_error',  # Verlustfunktion
    metrics=['mae']             # Mittlerer absoluter Fehler als zusätzliche Metrik
)
```

# Trainieren unser Modell
```python
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./Modells/C&F.keras',  # Speicherort des besten Modells
        save_best_only=True,            # Speichert nur das beste Modell
        monitor='val_loss',             # Überwache den Validierungsverlust
        mode='min'                      # Minimierung des Loss
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',             # Überwache den Validierungsverlust
        patience=10,                    # Stoppt, wenn sich der Verlust 10 Epochen nicht verbessert
        restore_best_weights=True       # Stellt die besten Gewichte wieder her
    )
]

history = model.fit(
    celcius_q_train, fahrenheit_a_train,
    validation_data=(celcius_q_label, fahrenheit_a_label),
    epochs=500,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

loss, mae = model.evaluate(celcius_q_label, fahrenheit_a_label, verbose=0)
print(f"Test Loss: {loss}, Test MAE: {mae}")
```

# Testen unser Modell
```python
print(model.predict(np.array([100.0])))
```

# Weights des Modells

```python
print("These are the layer variables: {}".format(model.get_weights()))
```