Modèle de prévision des prix du pétrole brut Brent
--------------------------------------------------
Ce script implémente un modèle d'ensemble pour prévoir les prix quotidiens 
du pétrole brut Brent en utilisant des indicateurs techniques et du machine learning.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import ta
import shap
import warnings
warnings.filterwarnings('ignore')

# Configuration
SEED = 42
np.random.seed(SEED)
MODEL_DIR = 'models'
FIGURE_DIR = 'figures'

# Création des répertoires pour sauvegarder les modèles et figures
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

def fetch_data(start_date='2000-01-01', end_date=None):
    """
    Récupère les données historiques du pétrole brut Brent via Yahoo Finance.
    
    Args:
        start_date (str): Date de début au format 'YYYY-MM-DD'
        end_date (str): Date de fin au format 'YYYY-MM-DD'
    
    Returns:
        pd.DataFrame: DataFrame contenant les données de prix
    """
    print(f"Récupération des données du Brent de {start_date} à aujourd'hui...")
    
    # Téléchargement des données
    data = yf.download("BZ=F", start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError("Aucune donnée n'a été récupérée. Vérifiez votre connexion internet ou le ticker.")
    
    print(f"Données récupérées: {len(data)} observations de {data.index.min()} à {data.index.max()}")
    
    # Préparation du DataFrame - FIX: s'assurer que l'index de date est correctement géré
    df = data.reset_index()
    
    # S'assurer que la colonne Close est renommée en Price
    if 'Close' in df.columns:
        df['Price'] = df['Close']  # Créer une nouvelle colonne Price au lieu de renommer
    
    return df

def create_features(df):
    """
    Crée les indicateurs techniques à partir des prix.
    
    Args:
        df (pd.DataFrame): DataFrame avec les colonnes 'Date' et 'Price'
    
    Returns:
        pd.DataFrame: DataFrame avec les features ajoutées
    """
    print("Création des indicateurs techniques...")
    
    # Assurer que les prix sont numériques et dans un format 1D approprié
    if 'Price' not in df.columns and 'Close' in df.columns:
        df['Price'] = df['Close'].astype(float)
    else:
        # S'assurer que Price est une série 1D (pas un DataFrame ou un array 2D)
        df['Price'] = df['Price'].values.flatten().astype(float)
    
    # S'assurer que les données sont triées par date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Moyennes mobiles
    df["ma_5"] = df["Price"].rolling(window=5).mean()
    df["ma_10"] = df["Price"].rolling(window=10).mean()
    df["ma_20"] = df["Price"].rolling(window=20).mean()
    
    # RSI
    df["rsi_14"] = ta.momentum.RSIIndicator(close=df["Price"], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(close=df["Price"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    
    # Bollinger Bands
    boll = ta.volatility.BollingerBands(close=df["Price"], window=20, window_dev=2)
    df["bb_upper"] = boll.bollinger_hband()
    df["bb_lower"] = boll.bollinger_lband()
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    
    # Momentum
    df["momentum_5"] = df["Price"] - df["Price"].shift(5)
    
    # Valeurs retardées (Lagged)
    for lag in range(1, 6):
        df[f"lag_{lag}"] = df["Price"].shift(lag)
    
    # Cible
    df['target'] = df['Price'].shift(-1)
    
    # Supprimer les lignes avec des valeurs manquantes
    df.dropna(inplace=True)
    
    return df

def analyze_correlations(df, features, target='target'):
    """
    Analyse et visualise les corrélations entre les features et la cible.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        features (list): Liste des noms de colonnes des features
        target (str): Nom de la colonne cible
    """
    print("Analyse des corrélations...")
    
    # Calcul de la matrice de corrélation
    corr_matrix = df[features + [target]].corr()
    
    # Extraction des corrélations avec la cible et tri
    correlations_with_target = corr_matrix[target].drop(target).sort_values(ascending=False)
    
    # Affichage des corrélations dans la console
    print("\nTop features par corrélation avec le prix du lendemain:")
    for feature, corr in correlations_with_target.head(10).items():
        print(f"{feature}: {corr:.4f}")
    
    # Visualisation de la matrice de corrélation
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f', cbar_kws={"shrink": .5})
    plt.title('Matrice de corrélation des features', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/correlation_matrix.png", dpi=300)
    
    # Graphique des corrélations avec la cible
    plt.figure(figsize=(12, 6))
    correlations_with_target.plot(kind='bar', color='skyblue')
    plt.title('Corrélation entre les features et le prix du lendemain', fontsize=16)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/target_correlations.png", dpi=300)

def train_models(X_train, y_train, X_train_scaled, X_test, y_test, X_test_scaled):
    """
    Entraîne les modèles de base et le metamodèle.
    
    Args:
        X_train, y_train: Données d'entraînement
        X_train_scaled: Données d'entraînement normalisées
        X_test, y_test: Données de test
        X_test_scaled: Données de test normalisées
        
    Returns:
        tuple: Modèles entraînés (rf, xgb, nn, metamodel) et les prédictions
    """
    print("Entraînement des modèles...")
    
    # 1. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds_train = rf.predict(X_train)
    rf_preds_test = rf.predict(X_test)
    
    # 2. XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=SEED, n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_preds_train = xgb.predict(X_train)
    xgb_preds_test = xgb.predict(X_test)
    
    # 3. Neural Network
    nn = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    nn.compile(optimizer='adam', loss='mse')
    nn.fit(X_train_scaled, y_train, epochs=50, batch_size=16, verbose=0)
    nn_preds_train = nn.predict(X_train_scaled).flatten()
    nn_preds_test = nn.predict(X_test_scaled).flatten()
    
    # Stacking pour le méta-modèle
    stacked_X_train = np.vstack([rf_preds_train, xgb_preds_train, nn_preds_train]).T
    stacked_X_test = np.vstack([rf_preds_test, xgb_preds_test, nn_preds_test]).T
    
    # Méta-modèle (régression linéaire)
    metamodel = LinearRegression()
    metamodel.fit(stacked_X_train, y_train)
    
    # Prédictions finales
    final_preds = metamodel.predict(stacked_X_test)
    
    # Évaluation
    mse = mean_squared_error(y_test, final_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, final_preds)
    
    print(f"\nPerformance du modèle d'ensemble:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Comparaison des modèles individuels
    print("\nPerformance des modèles individuels:")
    for name, preds in [("Random Forest", rf_preds_test), 
                        ("XGBoost", xgb_preds_test), 
                        ("Neural Network", nn_preds_test)]:
        model_rmse = np.sqrt(mean_squared_error(y_test, preds))
        model_r2 = r2_score(y_test, preds)
        print(f"{name}: RMSE = {model_rmse:.4f}, R² = {model_r2:.4f}")
    
    # Calcul des résidus
    residuals = y_test - final_preds
    
    return rf, xgb, nn, metamodel, final_preds, residuals

def visualize_results(y_test, final_preds, residuals, df_test, features):
    """
    Visualise les résultats du modèle.
    
    Args:
        y_test: Valeurs réelles
        final_preds: Prédictions du modèle
        residuals: Résidus
        df_test: DataFrame de test avec dates
        features: Liste des features
    """
    print("Visualisation des résultats...")
    
    # 1. Prix réels vs prédits
    plt.figure(figsize=(14, 7))
    plt.plot(df_test['Date'], y_test.values, label='Prix réels', color='blue', alpha=0.7)
    plt.plot(df_test['Date'], final_preds, label='Prix prédits', color='red', alpha=0.7)
    plt.title('Prix réels vs. Prix prédits du pétrole brut Brent', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Prix (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/actual_vs_predicted.png", dpi=300)
    
    # 2. Analyse des résidus
    plt.figure(figsize=(14, 7))
    plt.plot(df_test['Date'], residuals, color='green', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Résidus (erreurs de prédiction)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Erreur (USD)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/residuals.png", dpi=300)
    
    # 3. Distribution des résidus
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Distribution des résidus', fontsize=16)
    plt.axvline(x=0, color='r', linestyle='-')
    plt.xlabel('Erreur (USD)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/residuals_distribution.png", dpi=300)
    
    # 4. RMSE glissante
    window = 30  # Fenêtre de 30 jours
    squared_errors = (y_test.values - final_preds) ** 2
    rolling_mse = pd.Series(squared_errors).rolling(window=window).mean()
    rolling_rmse = np.sqrt(rolling_mse)
    
    plt.figure(figsize=(14, 7))
    plt.plot(df_test['Date'][window-1:], rolling_rmse[window-1:], color='purple', alpha=0.7)
    plt.title(f'RMSE glissante (fenêtre de {window} jours)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('RMSE (USD)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/rolling_rmse.png", dpi=300)

def analyze_feature_importance(xgb, X_test, features):
    """
    Analyse et visualise l'importance des features avec SHAP.
    
    Args:
        xgb: Modèle XGBoost entraîné
        X_test: Données de test
        features: Liste des noms de features
    """
    print("Analyse de l'importance des features avec SHAP...")
    
    # Calcul des valeurs SHAP pour XGBoost
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)
    
    # Graphique de synthèse (beeswarm plot)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
    plt.title('Impact des features sur la prédiction (SHAP values)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/shap_summary.png", dpi=300)
    
    # Graphique d'importance des features
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_test, feature_names=features, plot_type="bar", show=False)
    plt.title('Importance globale des features (SHAP values)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/shap_importance.png", dpi=300)

def save_models(rf, xgb, nn, metamodel, scaler):
    """
    Sauvegarde les modèles entrainés.
    
    Args:
        rf, xgb, nn, metamodel: Modèles entraînés
        scaler: StandardScaler ajusté
    """
    print("Sauvegarde des modèles...")
    
    joblib.dump(rf, f'{MODEL_DIR}/brent_rf_model.pkl')
    joblib.dump(xgb, f'{MODEL_DIR}/brent_xgb_model.pkl')
    joblib.dump(metamodel, f'{MODEL_DIR}/brent_metamodel.pkl')
    joblib.dump(scaler, f'{MODEL_DIR}/brent_scaler.pkl')
    nn.save(f'{MODEL_DIR}/brent_nn_model.h5')
    
    print(f"Modèles sauvegardés dans le dossier '{MODEL_DIR}'")

def load_models():
    """
    Charge les modèles sauvegardés.
    
    Returns:
        tuple: Modèles chargés (rf, xgb, nn, metamodel, scaler)
    """
    print("Chargement des modèles...")
    
    rf = joblib.load(f'{MODEL_DIR}/brent_rf_model.pkl')
    xgb = joblib.load(f'{MODEL_DIR}/brent_xgb_model.pkl')
    metamodel = joblib.load(f'{MODEL_DIR}/brent_metamodel.pkl')
    scaler = joblib.load(f'{MODEL_DIR}/brent_scaler.pkl')
    nn = load_model(f'{MODEL_DIR}/brent_nn_model.h5')
    
    return rf, xgb, nn, metamodel, scaler

def predict_next_day(df, features, rf, xgb, nn, metamodel, scaler):
    """
    Prédit le prix du pétrole Brent pour le jour suivant.
    
    Args:
        df (pd.DataFrame): DataFrame avec les données historiques
        features (list): Liste des features
        rf, xgb, nn, metamodel: Modèles entraînés
        scaler: StandardScaler ajusté
    
    Returns:
        float: Prix prédit pour le jour suivant
    """
    print("Prédiction du prix pour le jour suivant...")
    
    # Extraire les features les plus récentes
    latest_features = df[features].iloc[-1:].copy()
    latest_date = df['Date'].iloc[-1]
    current_price = df['Price'].iloc[-1]
    
    # Normaliser pour le réseau de neurones
    latest_scaled = scaler.transform(latest_features)
    
    # Prédire avec chaque modèle
    rf_pred = rf.predict(latest_features)[0]
    xgb_pred = xgb.predict(latest_features)[0]
    nn_pred = float(nn.predict(latest_scaled)[0])
    
    # Combiner avec le méta-modèle
    stacked_latest = np.array([[rf_pred, xgb_pred, nn_pred]])
    final_prediction = metamodel.predict(stacked_latest)[0]
    
    # Calcul du changement prévu
    change = final_prediction - current_price
    percent_change = (change / current_price) * 100
    
    print(f"\nDate actuelle: {latest_date}")
    print(f"Prix actuel: ${current_price:.2f}")
    print(f"Prix prédit pour le jour suivant: ${final_prediction:.2f}")
    print(f"Changement prévu: ${change:.2f} ({percent_change:.2f}%)")
    
    return final_prediction

def main():
    """Fonction principale exécutant le workflow complet."""
    print("\n" + "="*80)
    print("MODÈLE DE PRÉVISION DES PRIX DU PÉTROLE BRUT BRENT")
    print("="*80 + "\n")
    
    try:
        # 1. Récupération des données
        df = fetch_data(start_date='2000-01-01')
        
        # 2. Création des features
        df = create_features(df)
        
        # 3. Définition des features
        features = [
            'ma_5', 'ma_10', 'ma_20', 
            'rsi_14', 
            'macd', 'macd_signal', 'macd_diff',
            'bb_upper', 'bb_lower', 'bb_width', 
            'momentum_5',
            'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5'
        ]
        
        # 4. Analyse des corrélations
        analyze_correlations(df, features)
        
        # 5. Préparation des données
        X = df[features]
        y = df['target']
        
        # Division en ensembles d'entraînement et de test (20% des données les plus récentes)
        train_size = int(0.8 * len(df))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        df_test = df.iloc[train_size:].reset_index(drop=True)
        
        # Normalisation pour le réseau de neurones
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 6. Entraînement des modèles
        rf, xgb, nn, metamodel, final_preds, residuals = train_models(
            X_train, y_train, X_train_scaled, X_test, y_test, X_test_scaled
        )
        
        # 7. Visualisation des résultats
        visualize_results(y_test, final_preds, residuals, df_test, features)
        
        # 8. Analyse de l'importance des features
        analyze_feature_importance(xgb, X_test, features)
        
        # 9. Sauvegarde des modèles
        save_models(rf, xgb, nn, metamodel, scaler)
        
        # 10. Prédiction pour le jour suivant
        predict_next_day(df, features, rf, xgb, nn, metamodel, scaler)
        
    except Exception as e:
        print(f"\nErreur lors de l'exécution du script: {str(e)}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "="*80)
    print("PROCESSUS TERMINÉ - CONSULTEZ LES RÉSULTATS DANS LE DOSSIER 'figures'")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
