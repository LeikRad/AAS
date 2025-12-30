import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# Configuração para POUCOS DADOS
IMG_HEIGHT = 64  # REDUZIDO para modelo mais simples
IMG_WIDTH = 64
BATCH_SIZE = 8   # Pequeno para poucos dados
EPOCHS = 50

class BinaryAnomalyDetector:
    def __init__(self, img_height=64, img_width=64):
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.history = None
        
    def build_model(self):
        """Modelo SIMPLIFICADO para poucos dados"""
        model = keras.Sequential([
            # Apenas 1 camada convolucional
            layers.Conv2D(16, (3, 3), activation='relu', 
                         input_shape=(self.img_height, self.img_width, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Camadas densas mínimas
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # LR menor
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(),
                    keras.metrics.Recall(),
                    keras.metrics.AUC()]
        )
        
        self.model = model
        return model
    
    def train(self, train_ds, val_ds, class_weight=None):
        """Treina com opção de class_weight para balanceamento"""
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,  # Maior paciência para poucos dados
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7
        )
        
        checkpoint = keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',  # Mudado para val_loss
            save_best_only=True,
            verbose=1
        )
        
        self.history = self.model.fit(
            train_ds,
            epochs=EPOCHS,
            validation_data=val_ds,
            class_weight=class_weight,  # Suporte para balanceamento
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, test_ds):
        results = self.model.evaluate(test_ds, verbose=0)
        metrics = dict(zip(self.model.metrics_names, results))
        
        print("\n=== Resultados da Avaliação ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def predict(self, image):
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        prediction = self.model.predict(image, verbose=0)[0][0]
        is_anomaly = prediction > 0.5
        
        return {
            'probability': float(prediction),
            'is_anomaly': bool(is_anomaly),
            'label': 'ANOMALIA' if is_anomaly else 'NORMAL'
        }
    
    def plot_training_history(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(self.history.history['loss'], label='Treino')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validação')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.history.history['accuracy'], label='Treino')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Validação')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(self.history.history['precision'], label='Treino')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validação')
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(self.history.history['recall'], label='Treino')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validação')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath='anomaly_detector.keras'):
        self.model.save(filepath)
        print(f"Modelo salvo em: {filepath}")
    
    def load_model(self, filepath='anomaly_detector.keras'):
        self.model = keras.models.load_model(filepath)
        print(f"Modelo carregado de: {filepath}")


def prepare_datasets(data_dir, img_height=64, img_width=64, batch_size=8, 
                     validation_split=0.2, seed=42):
    """Prepara datasets com tamanhos corretos"""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale',
        label_mode='binary',
        shuffle=True
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale',
        label_mode='binary',
        shuffle=True
    )
    
    # Normalização
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Otimizações
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds


def prepare_test_dataset(test_dir, img_height=64, img_width=64, batch_size=8):
    """Prepara dataset de teste"""
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale',
        label_mode='binary',
        shuffle=False
    )
    
    normalization_layer = layers.Rescaling(1./255)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
    
    AUTOTUNE = tf.data.AUTOTUNE
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return test_ds


def add_data_augmentation(train_ds):
    """Data augmentation AGRESSIVO para poucos dados"""
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.4),  # Aumentado
        layers.RandomZoom(0.3),
        layers.RandomContrast(0.3),
        layers.RandomTranslation(0.2, 0.2),
    ])
    
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return train_ds


def calculate_class_weights(train_ds):
    """Calcula pesos para balancear classes"""
    all_labels = []
    for _, labels in train_ds:
        all_labels.extend(labels.numpy())
    
    all_labels = np.array(all_labels)
    total = len(all_labels)
    
    count_class_0 = np.sum(all_labels == 0)
    count_class_1 = np.sum(all_labels == 1)
    
    # Calcula pesos inversamente proporcionais
    weight_class_0 = total / (2 * count_class_0) if count_class_0 > 0 else 1.0
    weight_class_1 = total / (2 * count_class_1) if count_class_1 > 0 else 1.0
    
    class_weight = {0: weight_class_0, 1: weight_class_1}
    
    print(f"\n=== Distribuição de Classes ===")
    print(f"Classe 0 (Normal): {count_class_0} ({count_class_0/total*100:.1f}%)")
    print(f"Classe 1 (Anomaly): {count_class_1} ({count_class_1/total*100:.1f}%)")
    print(f"Class weights: {class_weight}")
    
    return class_weight


def print_dataset_info(data_dir):
    """Mostra informações sobre o dataset"""
    print("\n=== Informações do Dataset ===")
    
    normal_dir = os.path.join(data_dir, 'normal')
    anomaly_dir = os.path.join(data_dir, 'anomaly')
    
    normal_count = len([f for f in os.listdir(normal_dir) 
                       if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    anomaly_count = len([f for f in os.listdir(anomaly_dir) 
                        if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    total = normal_count + anomaly_count
    
    print(f"Imagens NORMAIS: {normal_count} ({normal_count/total*100:.1f}%)")
    print(f"Imagens ANOMALIAS: {anomaly_count} ({anomaly_count/total*100:.1f}%)")
    print(f"TOTAL: {total}")
    
    if total < 100:
        print("\n⚠️  AVISO: Menos de 100 imagens!")
        print("   Recomendação: Mínimo 200-500 imagens para CNN simples")
    elif total < 500:
        print("\n⚠️  Dataset pequeno detectado")
        print("   Modelo simplificado e augmentation agressivo serão usados")
    else:
        print("\n✅ Dataset adequado para CNN")
    
    return normal_count, anomaly_count


if __name__ == "__main__":
    DATA_DIR = "data_dir"
    TEST_DIR = "data_test"
    
    # Verificar informações do dataset
    print_dataset_info(DATA_DIR)
    
    print("\n=== Preparando Datasets ===")
    train_ds, val_ds = prepare_datasets(
        DATA_DIR, 
        img_height=IMG_HEIGHT, 
        img_width=IMG_WIDTH,
        batch_size=BATCH_SIZE,
        validation_split=0.2
    )
    
    # Calcular class weights para balanceamento
    class_weight = calculate_class_weights(train_ds)
    
    print("\n=== Aplicando Data Augmentation ===")
    train_ds = add_data_augmentation(train_ds)
    
    print("\n=== Construindo Modelo ===")
    detector = BinaryAnomalyDetector(IMG_HEIGHT, IMG_WIDTH)
    detector.build_model()
    
    print("\n=== Arquitetura do Modelo ===")
    detector.model.summary()
    
    print("\n=== Iniciando Treino ===")
    detector.train(train_ds, val_ds, class_weight=class_weight)
    
    if os.path.exists(TEST_DIR):
        print("\n=== Preparando Dataset de Teste ===")
        test_ds = prepare_test_dataset(TEST_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)
        
        print("\n=== Avaliação no Conjunto de Teste ===")
        detector.evaluate(test_ds)
    else:
        print("\n=== Avaliação no Conjunto de Validação ===")
        detector.evaluate(val_ds)
    
    print("\n=== Gerando Gráficos ===")
    detector.plot_training_history()
    
    print("\n=== Salvando Modelo ===")
    detector.save_model('binary_anomaly_detector.keras')
    
    print("\n=== Treino Concluído! ===")