from loguru import logger
import joblib
from processing.DataManager import (
    load_data,
    replace_nan,
    lower_columns,
    retain_first_cabin,
    add_titles,
    type_conversion,
    fill_by_median,
)
from processing.Prediction import (
    add_rare_status,
    process_categorical_data,
    scale_data,
    train_logistic_regression,
    evaluate_model
)
from sklearn.model_selection import train_test_split

def main():

    logger.add("logfile.log", rotation="500 MB")  

    logger.info("Début du pipeline Titanic")

    # Charger les données
    try:
        data = load_data('data/train.csv')  # Remplace par ton chemin de fichier
        logger.info("Données chargées avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données : {e}")
        return

    # Traitement des données
    try:
        data = replace_nan(data)
        data = lower_columns(data)
        data = retain_first_cabin(data)
        data = add_titles(data)
        logger.info("Traitement des données terminé")
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement des données : {e}")
        return
    
    target = 'survived'  # Le nom de la variable cible

    # Conversion des types de données
    try:
        vars_num, vars_cat = type_conversion(data, target)
        logger.info(f"Variables numériques : {vars_num}")
        logger.info(f"Variables catégorielles : {vars_cat}")
    except Exception as e:
        logger.error(f"Erreur lors de la conversion des types : {e}")
        return

    # Séparation des données en train et test
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(target, axis=1),  # Variables explicatives
            data[target],  # Variable cible
            test_size=0.2,  # Proportion de test
            random_state=0  # Pour la reproductibilité
        )
        logger.info(f"Jeux de données séparés : {X_train.shape[0]} en train et {X_test.shape[0]} en test")
    except Exception as e:
        logger.error(f"Erreur lors de la séparation des données : {e}")
        return

    # Remplir les valeurs manquantes et traiter les variables catégorielles
    try:
        X_train = fill_by_median(X_train)
        X_test = fill_by_median(X_test)
        logger.info("Valeurs manquantes traitées")
    except Exception as e:
        print(vars_cat)
        logger.error(f"Erreur lors du remplissage des valeurs manquantes : {e}")
        return

    
    try:
        X_train, X_test, variables = add_rare_status(X_train, X_test, vars_cat)
        logger.info("Variables catégorielles traitées et encodées")
    except Exception as e:
        logger.error(f"Erreur lors du traitement des variables catégorielles : {e}")
        return

    # Traitement des variables catégorielles : rareté et encoding
    try:
        X_train, X_test, variables = process_categorical_data(X_train, X_test, vars_cat)
        logger.info("Variables catégorielles traitées et encodées")
    except Exception as e:
        logger.error(f"Erreur lors du traitement des variables catégorielles : {e}")
        return

        

    # Mise à l'échelle des données
    try:
        X_train, X_test, scaler = scale_data(X_train, X_test, variables)
        logger.info("Données mises à l'échelle")
    except Exception as e:
        logger.error(f"Erreur lors de la mise à l'échelle des données : {e}")
        return

    # Entraînement du modèle
    try:
        model = train_logistic_regression(X_train, y_train)
        logger.info("Modèle entraîné avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement du modèle : {e}")
        return

    # Évaluation du modèle
    try:
        evaluate_model(model, X_train, y_train, X_test, y_test)
        logger.info("Évaluation du modèle terminée")
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation du modèle : {e}")
        return

    # Sauvegarder le modèle et le scaler
    try:
        joblib.dump(model, 'titanic_logistic_regression_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        logger.info("Modèle et scaler sauvegardés avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du modèle et du scaler : {e}")

if __name__ == '__main__':
    main()
