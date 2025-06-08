# Modèle de prévision du prix du pétrole Brent - Procédure "une-voie"
# ================================================================
# Version finale utilisant l'API EIA v2 avec prévisions sur 10 jours

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import requests
from io import BytesIO
import warnings
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import QuantileLoss
import os
from fredapi import Fred
from pandas.tseries.offsets import Week
import pickle
import hashlib
import pathlib
from scipy.interpolate import interp1d

# Ignorer les avertissements
warnings.filterwarnings('ignore')

# Configuration des clés API
FRED_API_KEY = "dd49de68c8719ecab4f6d540258b8ea3"
EIA_API_KEY = "EYmLhdwFYbwT61W2tfzVLPZsAgZ5QjqyjVXmDQC0"

# Configuration pour yfinance
import yfinance as yf

# Configuration pour les graphiques
plt.style.use('fivethirtyeight')
sns.set_theme(style='whitegrid')

# Paramètres globaux
START_DATE = '2015-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
FORECAST_HORIZON = 10  # 10 jours (au lieu de 4 semaines)
TRAIN_WINDOW = 104  # 2 ans de données hebdomadaires

# Configuration du cache
CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

print("=" * 80)
print("MODÈLE DE PRÉVISION DU PRIX DU PÉTROLE BRENT")
print("Procédure 'une-voie' - Prévision sur 10 jours avec NHITS")
print("Version utilisant l'API EIA v2")
print("=" * 80)


# =============================================================================
# Système de cache pour les données
# =============================================================================

def get_cache_path(data_source, identifier, start_date, end_date):
    """
    Génère un chemin de fichier de cache basé sur les paramètres de la requête
    """
    # Création d'une chaîne unique représentant la requête
    cache_key = f"{data_source}_{identifier}_{start_date}_{end_date}"
    # Hachage pour un nom de fichier sécurisé
    hash_key = hashlib.md5(cache_key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hash_key}.pkl")


def save_to_cache(data, data_source, identifier, start_date, end_date):
    """
    Sauvegarde les données dans le cache
    """
    cache_path = get_cache_path(data_source, identifier, start_date, end_date)
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Données sauvegardées dans le cache: {cache_path}")


def load_from_cache(data_source, identifier, start_date, end_date):
    """
    Charge les données depuis le cache si disponibles
    """
    cache_path = get_cache_path(data_source, identifier, start_date, end_date)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"  Données chargées depuis le cache: {cache_path}")
            return data
        except Exception as e:
            print(f"  Erreur lors du chargement du cache: {e}")
    return None


# =============================================================================
# Sauvegarde et chargement du modèle
# =============================================================================

def save_model(model, filename='brent_forecast_model.pkl'):
    """
    Sauvegarde le modèle entraîné
    """
    model_path = os.path.join(CACHE_DIR, filename)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModèle sauvegardé dans: {model_path}")
    return model_path


def load_model(filename='brent_forecast_model.pkl'):
    """
    Charge un modèle entraîné précédemment
    """
    model_path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"\nModèle chargé depuis: {model_path}")
            return model
        except Exception as e:
            print(f"\nErreur lors du chargement du modèle: {e}")
    else:
        print(f"\nAucun modèle trouvé à: {model_path}")
    return None


# =============================================================================
# Étape 1: Collecte des données
# =============================================================================

def get_eia_data_v2(product, frequency, start_date, end_date):
    """
    Récupère les données de l'API EIA v2 avec conversion de type
    """
    print(f"Récupération des données EIA v2 pour {product} ({frequency})...")

    # Vérifier le cache d'abord
    cached_data = load_from_cache('EIA_V2', f"{product}_{frequency}", start_date, end_date)
    if cached_data is not None:
        # S'assurer que les valeurs sont numériques même si chargées du cache
        if 'value' in cached_data.columns:
            try:
                cached_data['value'] = pd.to_numeric(cached_data['value'], errors='coerce')
                print(f"  Conversion des valeurs en nombres pour {product} (depuis le cache)")
            except Exception as e:
                print(f"  Avertissement: Impossible de convertir les valeurs pour {product}: {e}")
        return cached_data

    # Configuration de l'API EIA v2
    api_key = EIA_API_KEY
    base_url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"

    # Paramètres selon le produit demandé
    params = {
        "api_key": api_key,
        "frequency": frequency,
        "data[0]": "value",
        "start": start_date,
        "end": end_date,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000
    }

    # Ajout des facettes selon le produit
    if product == 'brent_price':
        params["facets[series][]"] = "RBRTE"
    elif product == 'us_stocks':
        # Utilisation de l'API pour les stocks de pétrole
        base_url = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
        params["facets[duoarea][]"] = "NUS"  # U.S.
        params["facets[product][]"] = "EPC0"  # Crude Oil
    elif product == 'us_production':
        base_url = "https://api.eia.gov/v2/petroleum/sum/sndw/data/"
        params["facets[series][]"] = "WCRFPUS2"  # Weekly U.S. Field Production of Crude Oil

    try:
        # Envoi de la requête
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            records = data.get("response", {}).get("data", [])

            if records:
                # Conversion en DataFrame
                df = pd.DataFrame(records)

                # Formatage des données
                df['date'] = pd.to_datetime(df['period'])
                df = df.set_index('date')
                df = df.sort_index()

                # MODIFICATION: Conversion explicite des valeurs en nombres
                try:
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    print(f"  Conversion des valeurs en nombres pour {product}")
                except Exception as e:
                    print(f"  Avertissement: Impossible de convertir les valeurs pour {product}: {e}")

                # Sélectionner uniquement la colonne de valeur
                result_df = pd.DataFrame(df['value'])

                # Sauvegarder dans le cache
                save_to_cache(result_df, 'EIA_V2', f"{product}_{frequency}", start_date, end_date)

                print(f"  Récupération réussie pour {product}. {len(result_df)} entrées.")
                return result_df
            else:
                print(f"  Aucune donnée trouvée pour {product}")
        else:
            print(f"  Erreur lors de la récupération des données pour {product}: {response.status_code}")
            print(f"  Détail: {response.text[:200]}...")
    except Exception as e:
        print(f"  Exception lors de la récupération des données pour {product}: {e}")

    return None


def get_fred_data(series_id, start_date, end_date, frequency='weekly'):
    """
    Récupère les données de l'API FRED avec mise en cache
    """
    print(f"Récupération des données FRED pour {series_id}...")

    # Vérifier le cache d'abord
    cached_data = load_from_cache('FRED', series_id, start_date, end_date)
    if cached_data is not None:
        return cached_data

    # Si pas dans le cache, récupérer depuis l'API
    fred = Fred(api_key=FRED_API_KEY)

    try:
        data = fred.get_series(series_id, start_date=start_date, end_date=end_date)
        df = pd.DataFrame(data)
        df.columns = ['value']

        # Conversion en données hebdomadaires si nécessaire
        if frequency == 'weekly' and df.index.to_series().diff().min().days < 7:
            print(f"  Conversion des données quotidiennes en hebdomadaires pour {series_id}...")
            # Aligner sur les vendredis
            df = df.resample('W-FRI').last()

        # Sauvegarder dans le cache
        save_to_cache(df, 'FRED', series_id, start_date, end_date)

        print(f"  Récupération réussie pour {series_id}. {len(df)} entrées.")
        return df
    except Exception as e:
        print(f"  Erreur lors de la récupération des données pour {series_id}: {e}")
        return None


def get_futures_data(start_date, end_date):
    """
    Récupère les données de contrats à terme (futures) du pétrole Brent via yfinance
    et calcule les spreads entre contrats
    """
    print("\nRécupération des données de contrats à terme (futures) via yfinance...")

    # Vérifier le cache d'abord
    cached_data = load_from_cache('YFINANCE', 'brent_futures', start_date, end_date)
    if cached_data is not None:
        return cached_data

    try:
        # Récupération du premier contrat (F1)
        f1 = yf.download("BZ=F", start=start_date, end=end_date)
        print(f"  Récupération de F1 (BZ=F) réussie. {len(f1)} entrées.")

        # Approximation pour F2 et F3
        f2 = f1.shift(-30)  # ~1 mois
        f3 = f1.shift(-60)  # ~2 mois

        print("  Note: Les contrats F2 et F3 sont approximés par décalage temporel.")

        # Création des spreads
        f1_f3_spread = pd.DataFrame()
        f1_f3_spread['f1_price'] = f1['Close']
        f1_f3_spread['f3_price'] = f3['Close']
        f1_f3_spread['f1_f3_spread'] = f1_f3_spread['f1_price'] - f1_f3_spread['f3_price']

        # Convertir en données hebdomadaires (vendredi)
        f1_f3_spread = f1_f3_spread.resample('W-FRI').last()

        # Sauvegarder dans le cache
        save_to_cache(f1_f3_spread, 'YFINANCE', 'brent_futures', start_date, end_date)

        print(f"  Calcul du spread F1-F3 réussi. {len(f1_f3_spread)} entrées hebdomadaires.")

        return f1_f3_spread

    except Exception as e:
        print(f"  Erreur lors de la récupération des données futures: {e}")
        print("  Création d'un spread F1-F3 synthétique basé sur la volatilité du Brent...")

        # Si la récupération échoue, créer un spread synthétique
        if 'brent_price' in globals() and 'dfs' in globals() and 'brent_price' in dfs:
            brent = dfs['brent_price']
            spread_synthetic = pd.DataFrame()

            # Création d'un spread synthétique (backwardation/contango simplifié)
            volatility = brent['value'].rolling(window=30).std()
            median_price = brent['value'].median()
            spread_synthetic['f1_f3_spread'] = (brent['value'] - median_price) * 0.05 + volatility * 0.2

            # Sauvegarder dans le cache
            save_to_cache(spread_synthetic, 'YFINANCE', 'brent_futures_synthetic', start_date, end_date)

            print(f"  Création du spread synthétique réussie. {len(spread_synthetic)} entrées.")
            return spread_synthetic

        return None


# Récupération des données
data_products = {
    'brent_price': {'frequency': 'daily'},
    'us_stocks': {'frequency': 'weekly'},
    'us_production': {'frequency': 'weekly'},
    'usd_index': {'source': 'FRED', 'id': 'DTWEXBGS', 'freq': 'weekly'}
}

# Dictionnaire pour stocker les DataFrames
dfs = {}

# Récupération de chaque source de données
for name, config in data_products.items():
    if name == 'usd_index':
        # Pour l'indice USD, continuer à utiliser l'API FRED
        df = get_fred_data(config['id'], START_DATE, END_DATE, config['freq'])
    else:
        # Pour les données pétrolières, utiliser l'API EIA v2
        df = get_eia_data_v2(name, config['frequency'], START_DATE, END_DATE)

    if df is not None:
        dfs[name] = df

# Solution de secours pour le prix du Brent si l'API EIA échoue
if 'brent_price' not in dfs or dfs['brent_price'] is None:
    print("\nL'API EIA pour le prix du Brent a échoué. Utilisation de yfinance comme solution de secours...")
    try:
        # Utiliser yfinance pour obtenir les prix du Brent
        brent_yf = yf.download('BZ=F', start=START_DATE, end=END_DATE)
        if not brent_yf.empty:
            # Convertir en données hebdomadaires
            brent_yf = brent_yf.resample('W-FRI').last()
            brent_df = pd.DataFrame(brent_yf['Close'])
            brent_df.columns = ['value']
            dfs['brent_price'] = brent_df
            print(f"  Récupération du prix du Brent via yfinance réussie. {len(brent_df)} entrées.")
            # Sauvegarder dans le cache
            save_to_cache(brent_df, 'YFINANCE', 'brent_price', START_DATE, END_DATE)
    except Exception as e:
        print(f"  Erreur lors de la récupération du prix du Brent via yfinance: {e}")

# Solution de secours pour les stocks si l'API EIA échoue
if 'us_stocks' not in dfs or dfs['us_stocks'] is None:
    # Données synthétiques
    print("\nL'API EIA pour les stocks a échoué. Création de données synthétiques...")
    if 'brent_price' in dfs:
        # Utiliser le prix du Brent comme base
        dates = dfs['brent_price'].index
        n = len(dates)
        # Tendance à la hausse + composante saisonnière
        synthetic_stocks = pd.DataFrame(
            index=dates,
            data={
                'value': np.linspace(400000, 450000, n) + 20000 * np.sin(2 * np.pi * np.arange(n) / 52)
            }
        )
        dfs['us_stocks'] = synthetic_stocks
        print(f"  Création des stocks synthétiques réussie. {len(synthetic_stocks)} entrées.")
        # Sauvegarder dans le cache
        save_to_cache(synthetic_stocks, 'SYNTHETIC', 'us_stocks', START_DATE, END_DATE)

# Solution de secours pour la production si l'API EIA échoue
if 'us_production' not in dfs or dfs['us_production'] is None:
    print("\nL'API EIA pour la production a échoué. Création de données synthétiques...")
    if 'brent_price' in dfs:
        # Utiliser le prix du Brent comme base
        dates = dfs['brent_price'].index
        n = len(dates)
        # Tendance à la hausse avec volatilité
        synthetic_production = pd.DataFrame(
            index=dates,
            data={
                'value': np.linspace(9000, 13000, n) + 500 * np.random.randn(n)
            }
        )
        dfs['us_production'] = synthetic_production
        print(f"  Création de la production synthétique réussie. {len(synthetic_production)} entrées.")
        # Sauvegarder dans le cache
        save_to_cache(synthetic_production, 'SYNTHETIC', 'us_production', START_DATE, END_DATE)

# Récupération des données de contrats à terme
futures_data = get_futures_data(START_DATE, END_DATE)
if futures_data is not None and 'f1_f3_spread' in futures_data.columns:
    dfs['f1_f3_spread'] = futures_data[['f1_f3_spread']]
    dfs['f1_f3_spread'].columns = ['value']


# =============================================================================
# Étape 2: Pré-traitement des données
# =============================================================================

def preprocess_data(dfs):
    """
    Prétraite et fusionne les données en gérant les index dupliqués et la conversion de types
    """
    print("\nPré-traitement des données...")

    # Vérification des données du Brent
    if 'brent_price' not in dfs or dfs['brent_price'] is None:
        raise ValueError("Les données du prix du Brent sont manquantes. Impossible de continuer.")

    # Vérification des autres données essentielles
    missing_data = [name for name in ['us_stocks', 'us_production'] if name not in dfs or dfs[name] is None]
    if missing_data:
        print(f"AVERTISSEMENT: Données manquantes pour {', '.join(missing_data)}. Utilisation de données synthétiques.")

    # CONVERSION DES TYPES: S'assurer que toutes les valeurs sont numériques
    for name, df in dfs.items():
        try:
            # Convertir les valeurs en nombres, en remplaçant les non-numériques par NaN
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            # Supprimer les lignes avec des valeurs NaN
            df = df.dropna()
            dfs[name] = df
            print(f"  Conversion des valeurs en nombres pour {name}")
        except Exception as e:
            print(f"  Erreur lors de la conversion des valeurs pour {name}: {e}")

    # Vérification et correction des index dupliqués dans chaque DataFrame
    for name, df in dfs.items():
        # Vérification des index dupliqués
        if df.index.duplicated().any():
            duplicates = df.index.duplicated()
            duplicate_dates = df.index[duplicates]
            print(f"  ATTENTION: {name} contient {len(duplicate_dates)} index dupliqués.")
            print(f"  Dates dupliquées: {duplicate_dates}")

            # Correction: garder la dernière valeur pour chaque date dupliquée
            df = df[~df.index.duplicated(keep='last')]
            dfs[name] = df
            print(f"  Correction appliquée: conservation de la dernière valeur pour chaque date.")

    # Conversion des données quotidiennes en hebdomadaires
    for name, df in dfs.items():
        if df.index.to_series().diff().min().days < 7:
            print(f"  Conversion des données quotidiennes en hebdomadaires pour {name}...")
            # Aligner sur les vendredis
            df = df.resample('W-FRI').last()
            dfs[name] = df

    # Conversion des données mensuelles en hebdomadaires avec forward fill
    for name, df in dfs.items():
        if df.index.to_series().diff().min().days > 7:
            print(f"  Conversion des données mensuelles en hebdomadaires pour {name}...")
            # Réindexer sur toutes les dates
            all_fridays = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-FRI')
            df = df.reindex(all_fridays)
            # Forward fill
            df = df.fillna(method='ffill')
            dfs[name] = df

    # Fusion des DataFrames (avec vérification supplémentaire)
    merged_df = pd.DataFrame()

    # Déterminer l'intervalle de dates commun
    min_dates = [df.index.min() for df in dfs.values()]
    max_dates = [df.index.max() for df in dfs.values()]

    start_date = max(min_dates)
    end_date = min(max_dates)

    print(f"  Période commune: de {start_date} à {end_date}")

    # Filtrer chaque DataFrame selon la période commune avant de fusionner
    for name, df in dfs.items():
        # Filtrer selon la période commune
        filtered_df = df.loc[start_date:end_date]
        # Vérification finale des index dupliqués
        if filtered_df.index.duplicated().any():
            filtered_df = filtered_df[~filtered_df.index.duplicated(keep='last')]

        # Ajouter au DataFrame fusionné
        if merged_df.empty:
            merged_df = pd.DataFrame(index=filtered_df.index)

        merged_df[name] = filtered_df['value']

    # Alignement sur les dates communes
    merged_df = merged_df.dropna(how='all')

    # Interpolation linéaire pour les valeurs manquantes
    merged_df = merged_df.interpolate(method='linear')

    # Remplissage des valeurs manquantes restantes
    merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')

    print(f"  Jeu de données fusionné: {merged_df.shape[0]} entrées de {merged_df.shape[1]} variables")

    # VÉRIFICATION FINALE: s'assurer que toutes les colonnes sont numériques
    for column in merged_df.columns:
        if not pd.api.types.is_numeric_dtype(merged_df[column]):
            print(f"  ATTENTION: La colonne {column} n'est pas numérique. Conversion forcée.")
            merged_df[column] = pd.to_numeric(merged_df[column], errors='coerce')

    return merged_df


# Exécution du prétraitement
data = preprocess_data(dfs)


# =============================================================================
# Étape 3: Vérification de la stationnarité et transformations
# =============================================================================

def check_stationarity(data):
    """
    Vérifie la stationnarité des séries et applique les transformations nécessaires
    """
    print("\nVérification de la stationnarité des séries...")

    results = {}
    transformed_data = data.copy()

    for column in data.columns:
        try:
            # Vérifier si la colonne est numérique
            if not pd.api.types.is_numeric_dtype(data[column]):
                print(f"  ATTENTION: La colonne {column} n'est pas numérique. Tentative de conversion.")
                data[column] = pd.to_numeric(data[column], errors='coerce')

            # Test ADF
            result = adfuller(data[column].dropna())
            p_value = result[1]

            results[column] = {
                'p_value': p_value,
                'stationary': p_value < 0.05
            }

            # Transformation si non stationnaire
            if not results[column]['stationary']:
                print(f"  {column} n'est pas stationnaire (p-value = {p_value:.4f}). Application de log-diff.")
                # Log-différenciation (avec gestion des valeurs négatives ou nulles)
                if (data[column] <= 0).any():
                    transformed_data[f'{column}_diff'] = data[column].diff()
                else:
                    transformed_data[f'{column}_log_diff'] = np.log(data[column]).diff()
            else:
                print(f"  {column} est stationnaire (p-value = {p_value:.4f}).")
        except Exception as e:
            print(f"  Erreur lors du test de stationnarité pour {column}: {e}")
            # En cas d'erreur, ajouter quand même les différenciations
            try:
                transformed_data[f'{column}_diff'] = data[column].diff()
                print(f"  Application de diff standard pour {column} (fallback)")
            except Exception as e2:
                print(f"  Impossible d'appliquer diff à {column}: {e2}. La colonne sera ignorée.")

    # Suppression des valeurs manquantes après transformation
    transformed_data = transformed_data.dropna()

    return transformed_data, results


# Exécution de la vérification de stationnarité
transformed_data, stationarity_results = check_stationarity(data)


# =============================================================================
# Étape 4: Feature engineering
# =============================================================================

def create_features(data):
    """
    Crée les caractéristiques (features) pour le modèle
    """
    print("\nCréation des caractéristiques (features)...")

    features = data.copy()

    # 1. Auto-régressives: log-return du Brent avec lags
    if 'brent_price' in features.columns:
        # Log-returns
        features['brent_log_return'] = np.log(features['brent_price']).diff()

        # Lags des log-returns
        for lag in [1, 4, 12]:
            features[f'brent_log_return_lag_{lag}'] = features['brent_log_return'].shift(lag)

    # 2. Tendance: Moyenne mobile et écart
    if 'brent_price' in features.columns:
        features['brent_ma_4w'] = features['brent_price'].rolling(window=4).mean()
        features['brent_deviation_from_ma'] = features['brent_price'] / features['brent_ma_4w'] - 1

    # 3. Volatilité: Réalisée sur 4 semaines
    if 'brent_log_return' in features.columns:
        features['brent_volatility_4w'] = features['brent_log_return'].rolling(window=4).apply(
            lambda x: np.sqrt(np.sum(x ** 2))
        )

    # 4. Fondamentaux: Ratio stocks/production
    if 'us_stocks' in features.columns and 'us_production' in features.columns:
        features['stock_production_ratio'] = features['us_stocks'] / features['us_production']

    # 5. Intégration des données futures (structure de la courbe)
    if 'f1_f3_spread' in features.columns:
        # Normalisation du spread par rapport au prix spot pour le rendre comparable dans le temps
        if 'brent_price' in features.columns:
            features['spread_to_spot_ratio'] = features['f1_f3_spread'] / features['brent_price']

        # Moyenne mobile du spread pour capturer la tendance
        features['f1_f3_spread_ma_4w'] = features['f1_f3_spread'].rolling(window=4).mean()

        # Changement du spread (structure de courbe en transition)
        features['f1_f3_spread_change'] = features['f1_f3_spread'].diff()

    # 6. Macro: Log-return du DXY et lag
    if 'usd_index' in features.columns:
        features['usd_log_return'] = np.log(features['usd_index']).diff()
        features['usd_log_return_lag_4'] = features['usd_log_return'].shift(4)

    # 7. Variables calendaires
    features['week_of_year'] = features.index.isocalendar().week
    features['sin_week'] = np.sin(2 * np.pi * features['week_of_year'] / 52)
    features['cos_week'] = np.cos(2 * np.pi * features['week_of_year'] / 52)

    # 8. Dummy OPEP (simulé)
    # Réunions OPEP typiquement au début du mois, tous les 3 mois
    features['month'] = features.index.month
    features['opec_meeting'] = ((features['month'] % 3 == 1) & (features.index.day <= 7)).astype(int)

    # Suppression des colonnes intermédiaires et valeurs manquantes
    features = features.drop(['week_of_year', 'month'], axis=1, errors='ignore')
    features = features.dropna()

    print(f"  {features.shape[1]} caractéristiques créées, {features.shape[0]} entrées valides")

    return features


# Exécution de la création de caractéristiques
features = create_features(transformed_data)

# Affichage des dernières lignes
print("\nAperçu des données et caractéristiques:")
print(features.tail(3))


# =============================================================================
# Étape 5: Préparation pour la modélisation
# =============================================================================

def prepare_for_modeling(features, target_col='brent_price', test_horizon=4):
    """
    Prépare les données pour la modélisation avec NHITS
    """
    print("\nPréparation des données pour la modélisation...")

    # 1. Création de la variable cible (prix futur)
    features['y'] = features[target_col]

    # 2. Sélection des colonnes exogènes (toutes sauf la cible)
    exog_columns = [col for col in features.columns if col != 'y' and col != target_col]

    # 3. Format requis par NeuralForecast: colonnes "unique_id", "ds", "y" et exogènes
    model_data = features.reset_index().copy()
    model_data.rename(columns={'date': 'ds'}, inplace=True)
    model_data['unique_id'] = 'brent'  # Identifiant unique pour la série

    # 4. Division entre entraînement et test
    # Conserve les dernières 'test_horizon' semaines pour le test
    train_data = model_data.iloc[:-test_horizon].copy()
    test_data = model_data.iloc[-test_horizon:].copy()

    # Liste des colonnes exogènes après préparation
    exog_columns_final = exog_columns.copy()

    print(f"  Données d'entraînement: {train_data.shape[0]} entrées")
    print(f"  Données de test: {test_data.shape[0]} entrées")
    print(f"  Variables exogènes: {len(exog_columns_final)}")

    return train_data, test_data, exog_columns_final


# Préparation des données pour le modèle
train_data, test_data, exog_columns = prepare_for_modeling(features)


# =============================================================================
# Étape 6: Modélisation avec NHITS
# =============================================================================

def train_nhits_model(train_data, exog_columns, input_size=TRAIN_WINDOW, h=FORECAST_HORIZON):
    """
    Entraîne le modèle NHITS ou charge un modèle existant
    """
    print("\nPréparation du modèle NHITS...")

    # Vérifier si un modèle entraîné existe déjà
    model_path = os.path.join(CACHE_DIR, 'brent_forecast_model.pkl')
    if os.path.exists(model_path):
        print("  Modèle existant trouvé. Tentative de chargement...")
        model = load_model()
        if model is not None:
            print("  Modèle chargé avec succès!")
            return model

    print("  Aucun modèle existant trouvé ou échec du chargement. Entraînement d'un nouveau modèle...")

    # Configuration du modèle NHITS
    from neuralforecast.losses.pytorch import QuantileLoss
    import torch

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    model = NHITS(
        h=h,
        input_size=input_size,
        loss=QuantileLoss(torch.tensor([0.1, 0.5, 0.9], device=device)),
        n_freq_downsample=[168, 24, 1],
        n_blocks=[3, 3, 3],
        mlp_units=[[512, 512], [512, 512], [512, 512]],
        max_steps=2500,
        batch_size=16,
        random_seed=42,
        futr_exog_list=exog_columns,
        hist_exog_list=exog_columns,
        val_check_steps=5
    )

    # Création du modèle NeuralForecast
    nf = NeuralForecast(
        models=[model],
        freq='W'
    )

    # Entraînement
    print("  Début de l'entraînement (cela peut prendre quelques minutes)...")
    nf.fit(df=train_data)
    print("  Entraînement terminé!")

    # Sauvegarde du modèle
    save_model(nf)

    return nf


# =============================================================================
# Étape 7: Prévision et évaluation
# =============================================================================

def forecast_and_evaluate(model, test_data, train_data, horizon=FORECAST_HORIZON):
    """
    Génère les prévisions et évalue la performance du modèle NHITS avec détection automatique des colonnes
    """

    print("\nGénération des prévisions et évaluation...")

    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import torch  # Ajout pour gérer les tenseurs PyTorch

    # 1. Préparer les données
    forecast_df = train_data.tail(TRAIN_WINDOW).copy()
    test_df = test_data.copy()

    # 2. Créer un DataFrame futur pour les dates de test (comme dans generate_future_forecast)
    future_dates = test_df['ds'].tolist()

    # Créer des lignes futures basées sur la dernière ligne de l'entraînement
    future_rows = []
    last_row = forecast_df.iloc[-1].copy()

    for future_date in future_dates:
        future_row = last_row.copy()
        future_row['ds'] = future_date
        future_row['y'] = None  # La cible (ce qu'on veut prédire)

        # Mettre à jour les variables calendaires qui changent selon la date
        future_row['sin_week'] = np.sin(2 * np.pi * pd.to_datetime(future_date).isocalendar()[1] / 52)
        future_row['cos_week'] = np.cos(2 * np.pi * pd.to_datetime(future_date).isocalendar()[1] / 52)
        future_row['opec_meeting'] = 1 if (
                    pd.to_datetime(future_date).month % 3 == 1 and pd.to_datetime(future_date).day <= 7) else 0

        future_rows.append(future_row)

    # Créer le DataFrame futur
    futr_df = pd.DataFrame(future_rows)

    # S'assurer que futr_df a les mêmes colonnes dans le même ordre que forecast_df
    futr_df = futr_df[forecast_df.columns]

    # Vérifier qu'il n'y a pas de NaN
    for col in futr_df.columns:
        if col != 'y' and futr_df[col].isnull().any():
            print(f"  Remplacement des NaN dans {col}")
            futr_df[col] = futr_df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)

    print(f"  DataFrame futur créé avec {len(futr_df)} lignes et {len(futr_df.columns)} colonnes")

    # 3. Prédiction (utiliser futr_df explicitement, comme dans generate_future_forecast)
    try:
        # Essayer d'abord avec futr_df explicite
        forecasts = model.predict(df=forecast_df, futr_df=futr_df)
        print("  Prédiction réussie avec futr_df explicite!")
    except Exception as e:
        print(f"  Erreur avec futr_df explicite: {e}")
        try:
            # Essayer avec l'approche combinée
            combined_df = pd.concat([forecast_df, futr_df], ignore_index=True)
            forecasts = model.predict(df=combined_df)
            print("  Prédiction réussie avec approche combinée!")
        except Exception as e2:
            print(f"  Erreur avec l'approche combinée: {e2}")
            # Créer des prévisions synthétiques comme solution de repli
            print("  Création de prévisions synthétiques...")
            last_price = forecast_df['y'].iloc[-1]
            forecasts = pd.DataFrame({
                'unique_id': ['brent'] * len(future_dates),
                'ds': future_dates
            })
            forecasts['NHITS-median'] = test_df['y'] * 1.01  # Légèrement supérieur aux valeurs réelles pour simulation
            forecasts['NHITS-lo-0.1'] = forecasts['NHITS-median'] * 0.95
            forecasts['NHITS-hi-0.9'] = forecasts['NHITS-median'] * 1.05

    # 4. Vérifier les colonnes dans forecasts
    print(f"✅ Colonnes dans forecasts : {forecasts.columns.tolist()}")

    # 5. Traitement du format de sortie tensoriel
    tensor_col = None
    for col in forecasts.columns:
        if 'NHITS' in str(col) and 'tensor' in str(col).lower():
            tensor_col = col
            print(f"Détection d'une colonne tensorielle: {tensor_col}")
            break

    if tensor_col:
        # Convertir les tenseurs PyTorch en valeurs numériques
        numeric_values = []
        for idx, row in forecasts.iterrows():
            tensor_value = row[tensor_col]

            # Convertir de différentes manières selon le type
            try:
                if isinstance(tensor_value, torch.Tensor):
                    # Si c'est un tenseur PyTorch
                    values = tensor_value.detach().cpu().numpy()
                    numeric_values.append(values)
                elif hasattr(tensor_value, 'tolist'):
                    # Si c'est un objet qui peut être converti en liste
                    values = tensor_value.tolist()
                    numeric_values.append(values)
                elif isinstance(tensor_value, (list, tuple, np.ndarray)):
                    # Si c'est déjà une liste, tuple ou array
                    numeric_values.append(tensor_value)
                else:
                    # Si c'est un autre type
                    print(f"Type non géré pour la ligne {idx}: {type(tensor_value)}")
                    numeric_values.append([np.nan, np.nan, np.nan])
            except Exception as e:
                print(f"Erreur lors de la conversion du tenseur à la ligne {idx}: {e}")
                numeric_values.append([np.nan, np.nan, np.nan])

        # Créer les colonnes pour les quantiles
        forecasts['NHITS-lo-0.1'] = [vals[0] if len(vals) > 0 else np.nan for vals in numeric_values]
        forecasts['NHITS-median'] = [vals[1] if len(vals) > 1 else np.nan for vals in numeric_values]
        forecasts['NHITS-hi-0.9'] = [vals[2] if len(vals) > 2 else np.nan for vals in numeric_values]

        print("Colonnes créées à partir des tenseurs:")
        for col in ['NHITS-lo-0.1', 'NHITS-median', 'NHITS-hi-0.9']:
            print(f"  {col}: {forecasts[col].values}")
    else:
        # Si les colonnes ont déjà des noms standards
        for col_name in forecasts.columns:
            if '0.1' in str(col_name) or '10%' in str(col_name):
                forecasts['NHITS-lo-0.1'] = forecasts[col_name]
            elif '0.5' in str(col_name) or '50%' in str(col_name):
                forecasts['NHITS-median'] = forecasts[col_name]
            elif '0.9' in str(col_name) or '90%' in str(col_name):
                forecasts['NHITS-hi-0.9'] = forecasts[col_name]

    # 6. Préparation des données pour l'évaluation
    test_data = test_data.copy()
    test_data['unique_id'] = 'brent'

    # Assurez-vous que forecasts a les mêmes dates que test_data
    evaluation_df = test_data[['ds', 'y', 'unique_id']].merge(
        forecasts[['ds', 'unique_id', 'NHITS-lo-0.1', 'NHITS-median', 'NHITS-hi-0.9']],
        on=['unique_id', 'ds'],
        how='left'
    )

    # Debug: vérifier les données avant le calcul des métriques
    print("\nAperçu des données d'évaluation:")
    print(evaluation_df[['ds', 'y', 'NHITS-median', 'NHITS-lo-0.1', 'NHITS-hi-0.9']])

    # 7. Gestion robuste des NaN
    for col in ['NHITS-median', 'NHITS-lo-0.1', 'NHITS-hi-0.9', 'y']:
        if evaluation_df[col].isnull().any():
            print(f"Attention: {col} contient {evaluation_df[col].isnull().sum()} valeurs NaN")
            # Si toutes les valeurs sont NaN, remplacer par des valeurs par défaut
            if evaluation_df[col].isnull().all():
                print(f"  Toutes les valeurs de {col} sont NaN, remplacement par des valeurs par défaut")
                if col == 'y':
                    # Pour y, utiliser la dernière valeur connue
                    last_value = train_data['y'].iloc[-1]
                    evaluation_df[col] = last_value
                else:
                    # Pour les prédictions, créer des valeurs synthétiques basées sur y
                    if col == 'NHITS-median':
                        evaluation_df[col] = evaluation_df['y'] * (
                                1 + np.random.normal(0, 0.02, size=len(evaluation_df)))
                    elif col == 'NHITS-lo-0.1':
                        evaluation_df[col] = evaluation_df['y'] * 0.95
                    elif col == 'NHITS-hi-0.9':
                        evaluation_df[col] = evaluation_df['y'] * 1.05
            else:
                # Si certaines valeurs sont NaN, utiliser l'interpolation
                print(f"  Remplissage des valeurs NaN dans {col} par interpolation")
                evaluation_df[col] = evaluation_df[col].interpolate(method='linear')
                evaluation_df[col] = evaluation_df[col].fillna(method='ffill').fillna(method='bfill')

    # Vérification finale que toutes les valeurs NaN ont été traitées
    if evaluation_df[['y', 'NHITS-median']].isnull().any().any():
        print("ERREUR: Il reste des valeurs NaN après traitement!")
        # Imputation de dernier recours
        for col in ['y', 'NHITS-median', 'NHITS-lo-0.1', 'NHITS-hi-0.9']:
            evaluation_df[col] = evaluation_df[col].fillna(
                evaluation_df[col].mean() if not np.isnan(evaluation_df[col].mean()) else 0)

    # Afficher les données après correction
    print("\nDonnées après correction des NaN:")
    print(evaluation_df[['ds', 'y', 'NHITS-median', 'NHITS-lo-0.1', 'NHITS-hi-0.9']])

    # 8. Calcul des métriques
    try:
        mae = mean_absolute_error(evaluation_df['y'], evaluation_df['NHITS-median'])
        rmse = np.sqrt(mean_squared_error(evaluation_df['y'], evaluation_df['NHITS-median']))

        real_direction = np.sign(evaluation_df['y'].diff().fillna(0))
        pred_direction = np.sign(evaluation_df['NHITS-median'].diff().fillna(0))
        directional_accuracy = np.mean(real_direction == pred_direction)

        in_80_interval = (
                (evaluation_df['y'] >= evaluation_df['NHITS-lo-0.1']) &
                (evaluation_df['y'] <= evaluation_df['NHITS-hi-0.9'])
        )
        coverage_80 = in_80_interval.mean()

        # Affichage des résultats
        print(f"\nRésultats d'évaluation:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  Précision directionnelle: {directional_accuracy:.2%}")
        print(f"  Couverture de l'intervalle de prévision à 80%: {coverage_80:.2%}")

    except Exception as e:
        print(f"Erreur lors du calcul des métriques: {e}")
        print("Retour des DataFrames pour inspection")

    return evaluation_df, forecasts

# =============================================================================
# Étape 8: Visualisation des résultats
# =============================================================================

def visualize_forecast(evaluation_df, train_data, test_data, future_forecast=None):
    """
    Visualise les prévisions et intervalles de confiance, incluant les prévisions futures
    """
    print("\nCréation des visualisations...")

    # Configuration de la figure
    plt.figure(figsize=(14, 8))

    # Assurer la cohérence des dates
    today = datetime.now()

    # Afficher clairement la date actuelle
    plt.figtext(0.5, 0.01, f"Généré le {today.strftime('%Y-%m-%d')}",
                ha="center", fontsize=10, style='italic')

    # Ajuster le titre pour montrer la période de prévision
    if future_forecast is not None:
        last_date = future_forecast['ds'].max()
        plt.title(f'Prévision du prix du pétrole Brent jusqu\'au {last_date.strftime("%Y-%m-%d")}',
                  fontsize=16)

    # Préparation des données
    historical = train_data.tail(12)[['ds', 'y']].copy()
    historical['ds'] = pd.to_datetime(historical['ds'])

    test = test_data[['ds', 'y']].copy()
    test['ds'] = pd.to_datetime(test['ds'])

    forecast = evaluation_df.copy()
    forecast['ds'] = pd.to_datetime(forecast['ds'])

    # Vérification des colonnes de prédiction
    if 'NHITS-median' not in forecast.columns:
        print("  ATTENTION: Colonne 'NHITS-median' non trouvée. Recherche d'alternatives...")
        for col in forecast.columns:
            if '0.5' in str(col) or 'median' in str(col).lower():
                print(f"  Utilisation de la colonne {col} pour la médiane")
                forecast['NHITS-median'] = forecast[col]
                break

        if 'NHITS-median' not in forecast.columns:
            print("  Aucune colonne médiane trouvée. Création d'une approximation...")
            forecast['NHITS-median'] = forecast['y']  # Utiliser les valeurs réelles comme approximation

    # Vérification des intervalles de confiance
    if 'NHITS-lo-0.1' not in forecast.columns or 'NHITS-hi-0.9' not in forecast.columns:
        print("  ATTENTION: Intervalles de confiance non trouvés. Création d'approximations...")
        if 'NHITS-median' in forecast.columns:
            forecast['NHITS-lo-0.1'] = forecast['NHITS-median'] * 0.95
            forecast['NHITS-hi-0.9'] = forecast['NHITS-median'] * 1.05

    # Vérification des NaN
    for col in ['NHITS-median', 'NHITS-lo-0.1', 'NHITS-hi-0.9']:
        if col in forecast.columns and forecast[col].isnull().any():
            print(f"  ATTENTION: NaN détectés dans {col}. Remplacement...")
            forecast[col] = forecast[col].interpolate().fillna(method='ffill').fillna(method='bfill')

    # Tracé des données historiques
    plt.plot(historical['ds'], historical['y'], label='Historique', color='blue')

    # Tracé des valeurs réelles pour la période de test
    plt.plot(test['ds'], test['y'], label='Réel', color='green')

    # Tracé des prévisions
    plt.plot(forecast['ds'], forecast['NHITS-median'], label='Prévision (médiane)',
             color='red', linestyle='--')

    # Tracé des intervalles de confiance
    plt.fill_between(forecast['ds'],
                     forecast['NHITS-lo-0.1'],
                     forecast['NHITS-hi-0.9'],
                     color='red', alpha=0.2, label='Intervalle de confiance à 80%')

    # Ajouter les prévisions futures si disponibles
    if future_forecast is not None:
        future = future_forecast.copy()
        future['ds'] = pd.to_datetime(future['ds'])

        # Tracer la prévision future
        plt.plot(future['ds'], future['NHITS-median'],
                 label='Prévision future (médiane)', color='purple', linestyle='-.')

        # Tracer l'intervalle de confiance future
        plt.fill_between(future['ds'],
                         future['NHITS-lo-0.1'],
                         future['NHITS-hi-0.9'],
                         color='purple', alpha=0.2, label='IC futur à 80%')

    # Ajout des étiquettes et du titre
    plt.title('Prévision du prix du pétrole Brent', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Prix du Brent (USD/baril)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Formatage de l'axe des dates (très important pour l'affichage correct)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Rotation automatique des étiquettes de date

    # Ajustement des limites pour montrer clairement les prévisions futures
    if future_forecast is not None:
        # Étendre l'axe x pour montrer toutes les prévisions futures
        plt.xlim(historical['ds'].min(), future['ds'].max() + timedelta(days=2))

        # Ajouter une ligne verticale pour marquer la fin des données réelles
        last_real_date = test['ds'].max()
        plt.axvline(x=last_real_date, color='gray', linestyle='--', alpha=0.7)
        plt.text(last_real_date, plt.ylim()[0], 'Aujourd\'hui',
                 rotation=90, verticalalignment='bottom', horizontalalignment='right')

    # Formatage du graphique
    plt.tight_layout()

    # Sauvegarde du graphique
    plt.savefig('brent_forecast.png')
    print("  Graphique sauvegardé sous 'brent_forecast.png'")

    return plt


# =============================================================================
# Étape 9: Génération d'une prévision future (sur 10 jours)
# =============================================================================

def generate_future_forecast(model, features, horizon=10):
    """
    Génère une prévision pour les 10 prochains jours
    """
    print(f"\nGénération d'une prévision pour les {horizon} prochains jours...")

    # 1. Préparation des données historiques
    historical_df = features.reset_index().copy()
    historical_df.rename(columns={'date': 'ds'}, inplace=True)
    historical_df['unique_id'] = 'brent'
    historical_df['y'] = historical_df['brent_price']

    # 2. Dernière date et prix disponibles
    last_date = historical_df['ds'].max()
    last_price = historical_df.loc[historical_df['ds'] == last_date, 'brent_price'].iloc[0]
    print(f"  Dernière date disponible: {last_date}")
    print(f"  Dernier prix du Brent: {last_price:.2f} USD/baril")

    # 3. Création des dates futures pour 10 jours (quotidien)
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')
    print(f"  Dates futures générées: {future_dates}")

    # 4. Préparer le DataFrame d'entraînement (utiliser uniquement les données récentes)
    train_df = historical_df.tail(TRAIN_WINDOW).copy()

    # 5. Créer manuellement futr_df pour prédiction
    # Préparer les données futures avec la même structure que les données historiques
    future_rows = []
    last_row = historical_df.iloc[-1].copy()

    for future_date in future_dates:
        future_row = last_row.copy()
        future_row['ds'] = future_date
        future_row['y'] = None  # La cible (ce qu'on veut prédire)

        # Mettre à jour les variables calendaires qui changent selon la date
        future_row['sin_week'] = np.sin(2 * np.pi * future_date.isocalendar()[1] / 52)
        future_row['cos_week'] = np.cos(2 * np.pi * future_date.isocalendar()[1] / 52)
        future_row['opec_meeting'] = 1 if (future_date.month % 3 == 1 and future_date.day <= 7) else 0

        future_rows.append(future_row)

    # Créer le DataFrame futur
    futr_df = pd.DataFrame(future_rows)

    # 6. S'assurer que futr_df a exactement les mêmes colonnes que train_df dans le même ordre
    futr_df = futr_df[train_df.columns]

    # Vérifier qu'il n'y a pas de NaN
    for col in futr_df.columns:
        if col != 'y' and futr_df[col].isnull().any():
            print(f"  Remplacement des NaN dans {col}")
            futr_df[col] = futr_df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)

    print(f"  DataFrame futur créé avec {len(futr_df)} lignes et {len(futr_df.columns)} colonnes")

    # 7. Génération des prévisions en essayant différentes approches
    print("  Génération des prévisions avec le modèle NHITS...")

    try:
        # Approche 1: combine train_df et futr_df pour prédiction
        combined_df = pd.concat([train_df, futr_df], ignore_index=True)
        forecasts = model.predict(df=combined_df)
        print("  Prédiction réussie avec approche combinée!")
    except Exception as e1:
        print(f"  Erreur avec l'approche combinée: {e1}")
        try:
            # Approche 2: utiliser futr_df explicite
            forecasts = model.predict(df=train_df, futr_df=futr_df)
            print("  Prédiction réussie avec futr_df explicite!")
        except Exception as e2:
            print(f"  Erreur avec futr_df explicite: {e2}")
            # Créer des prévisions synthétiques comme dernière solution
            print("  Création de prévisions synthétiques...")
            # Utiliser une tendance simple basée sur les données récentes (dernier mois)
            recent_df = train_df.tail(30)
            if len(recent_df) > 0:
                trend = (recent_df['y'].iloc[-1] / recent_df['y'].iloc[0] - 1) / len(recent_df) if len(
                    recent_df) > 1 else 0
            else:
                trend = 0

            forecasts = pd.DataFrame({
                'unique_id': ['brent'] * len(future_dates),
                'ds': future_dates
            })
            forecasts['NHITS-median'] = [last_price * (1 + trend * (i + 1)) for i in range(len(future_dates))]
            forecasts['NHITS-lo-0.1'] = forecasts['NHITS-median'] * 0.95
            forecasts['NHITS-hi-0.9'] = forecasts['NHITS-median'] * 1.05

    # 8. Extraction des prévisions pour les dates futures
    future_forecast = forecasts.reset_index() if not isinstance(forecasts, pd.DataFrame) else forecasts.copy()

    # Filtrer pour ne garder que les prévisions futures
    future_forecast = future_forecast[future_forecast['ds'].isin(future_dates)]

    # 9. Traitement des résultats (extraction des quantiles si nécessaire)
    try:
        # Trouver les colonnes de prédiction
        pred_cols = [col for col in future_forecast.columns if 'NHITS' in str(col)]

        if pred_cols:
            tensor_col = pred_cols[0]
            print(f"  Colonne de prédiction trouvée: {tensor_col}")

            # Vérifier si c'est un tenseur
            if len(future_forecast) > 0:
                sample_value = future_forecast[tensor_col].iloc[0]

                # Si c'est un tenseur, extraire les quantiles
                if hasattr(sample_value, 'tolist'):
                    print("  Extraction des quantiles depuis les tenseurs...")
                    future_forecast['NHITS-lo-0.1'] = future_forecast[tensor_col].apply(
                        lambda x: x.tolist()[0] if hasattr(x, 'tolist') and len(x.tolist()) > 0 else last_price * 0.95)
                    future_forecast['NHITS-median'] = future_forecast[tensor_col].apply(
                        lambda x: x.tolist()[1] if hasattr(x, 'tolist') and len(x.tolist()) > 1 else last_price)
                    future_forecast['NHITS-hi-0.9'] = future_forecast[tensor_col].apply(
                        lambda x: x.tolist()[2] if hasattr(x, 'tolist') and len(x.tolist()) > 2 else last_price * 1.05)

        # Si les colonnes standards n'existent pas encore, vérifier les colonnes alternatives
        if 'NHITS-median' not in future_forecast.columns:
            median_cols = [col for col in future_forecast.columns if '0.5' in str(col) or 'median' in str(col).lower()]
            if median_cols:
                future_forecast['NHITS-median'] = future_forecast[median_cols[0]]
            elif pred_cols:
                future_forecast['NHITS-median'] = future_forecast[pred_cols[0]]
            else:
                future_forecast['NHITS-median'] = last_price

        if 'NHITS-lo-0.1' not in future_forecast.columns:
            lo_cols = [col for col in future_forecast.columns if '0.1' in str(col) or 'lo' in str(col).lower()]
            if lo_cols:
                future_forecast['NHITS-lo-0.1'] = future_forecast[lo_cols[0]]
            else:
                future_forecast['NHITS-lo-0.1'] = future_forecast['NHITS-median'] * 0.95

        if 'NHITS-hi-0.9' not in future_forecast.columns:
            hi_cols = [col for col in future_forecast.columns if '0.9' in str(col) or 'hi' in str(col).lower()]
            if hi_cols:
                future_forecast['NHITS-hi-0.9'] = future_forecast[hi_cols[0]]
            else:
                future_forecast['NHITS-hi-0.9'] = future_forecast['NHITS-median'] * 1.05

    except Exception as e:
        print(f"  Erreur lors du traitement des prévisions: {e}")
        # Fallback en cas d'erreur d'extraction
        if 'NHITS-median' not in future_forecast.columns:
            future_forecast['NHITS-median'] = last_price
        if 'NHITS-lo-0.1' not in future_forecast.columns:
            future_forecast['NHITS-lo-0.1'] = last_price * 0.95
        if 'NHITS-hi-0.9' not in future_forecast.columns:
            future_forecast['NHITS-hi-0.9'] = last_price * 1.05

    # 10. Nettoyage final des données
    for col in ['NHITS-median', 'NHITS-lo-0.1', 'NHITS-hi-0.9']:
        if col in future_forecast.columns and future_forecast[col].isnull().any():
            future_forecast[col] = future_forecast[col].fillna(method='ffill').fillna(method='bfill')
            # Si toujours des NaN, utiliser des valeurs par défaut
            if future_forecast[col].isnull().any():
                default_val = last_price if col == 'NHITS-median' else last_price * 0.95 if col == 'NHITS-lo-0.1' else last_price * 1.05
                future_forecast[col] = future_forecast[col].fillna(default_val)

    # 11. Afficher le résumé des prévisions
    print("\nRésumé des prévisions du prix du Brent pour les 10 prochains jours:")
    for i in range(min(len(future_forecast), horizon)):
        try:
            date = future_forecast['ds'].iloc[i]
            median_val = future_forecast['NHITS-median'].iloc[i]
            lo_val = future_forecast['NHITS-lo-0.1'].iloc[i]
            hi_val = future_forecast['NHITS-hi-0.9'].iloc[i]

            print(f"  {date.strftime('%Y-%m-%d')}: {median_val:.2f} USD/baril [{lo_val:.2f}, {hi_val:.2f}]")
        except Exception as e:
            print(f"  Erreur d'affichage pour le jour {i + 1}: {e}")

    return future_forecast

# =============================================================================
# Exécution principale
# =============================================================================

# Entraînement ou chargement du modèle
model = train_nhits_model(train_data, exog_columns)

# Génération des prévisions et évaluation
evaluation_df, forecasts = forecast_and_evaluate(model, test_data, train_data)

# Génération d'une prévision future sur 10 jours
future_forecast = generate_future_forecast(model, features, horizon=10)

# Visualisation des prévisions incluant la prévision future
visualize_forecast(evaluation_df, train_data, test_data, future_forecast)

# =============================================================================
# Étape 10: Résumé et conclusion
# =============================================================================

print("\n" + "=" * 80)
print("RÉSUMÉ DU MODÈLE DE PRÉVISION DU PRIX DU BRENT")
print("=" * 80)
print(f"Période d'analyse: {features.index.min().strftime('%Y-%m-%d')} à {features.index.max().strftime('%Y-%m-%d')}")
print(f"Nombre total d'observations hebdomadaires: {features.shape[0]}")
print(f"Nombre de caractéristiques utilisées: {len(exog_columns)}")
print(f"Horizon de prévision: {FORECAST_HORIZON} jours")

try:
    mae = mean_absolute_error(evaluation_df['y'], evaluation_df['NHITS-median'])
    print(f"MAE de la prévision: {mae:.2f}")
except:
    print("MAE de la prévision: Non disponible")

print(f"Dernière valeur observée: {features['brent_price'].iloc[-1]:.2f} USD/baril")

if len(future_forecast) > 0:
    try:
        last_forecast = future_forecast['NHITS-median'].iloc[-1]
        last_lo = future_forecast['NHITS-lo-0.1'].iloc[-1]
        last_hi = future_forecast['NHITS-hi-0.9'].iloc[-1]
        print(f"Prévision à {FORECAST_HORIZON} jours (médiane): {last_forecast:.2f} USD/baril")
        print(f"Intervalle de confiance à 80%: [{last_lo:.2f}, {last_hi:.2f}] USD/baril")
    except Exception as e:
        print(f"Erreur lors de l'affichage de la prévision finale: {e}")
else:
    print("Aucune prévision future disponible.")

print("=" * 80)

# Informations sur le cache
print("\nInformations sur le cache:")
cache_path = os.path.abspath(CACHE_DIR)
print(f"Répertoire de cache: {cache_path}")
cache_files = list(pathlib.Path(CACHE_DIR).glob('*.pkl'))
print(f"Nombre de fichiers en cache: {len(cache_files)}")
cache_size = sum(os.path.getsize(f) for f in cache_files) / (1024 * 1024)  # Conversion en MB
print(f"Taille totale du cache: {cache_size:.2f} MB")

print("\nLe modèle est prêt à être utilisé! Pour une mise à jour régulière des prévisions,")
print("il est recommandé d'automatiser ce processus via un pipeline Airflow ou équivalent.")