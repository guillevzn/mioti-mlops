from xgboost import XGBClassifier
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import joblib

class BuyerClassification(BaseEstimator, TransformerMixin):
    def __init__(self, df_cat=None, onehot_encoder=None):
        self.model = None
        self.onehot_encoder = onehot_encoder

    def _create_onehot_encoder(self, df_cat):
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        onehot_encoder.fit(df_cat)
        return onehot_encoder
    
    def predict(self, X):
        X_processed = self._preprocess_data(X)
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)
        
        # Tomamos la probabilidad de la clase positiva
        positive_probabilities = probabilities[:, 1]
        
        return predictions, positive_probabilities

    def _preprocess_data(self, df):
        # Convertir 'event_time' a datetime
        df['event_time'] = pd.to_datetime(df['event_time'])

        # Crear el DataFrame `user_sessions`
        user_sessions = df.groupby(['user_id', 'user_session', 'category_1']).agg(
            num_events=('event_time', 'size'),
            session_start_time=('event_time', lambda x: x.min().timestamp()),
            session_end_time=('event_time', lambda x: x.max().timestamp()),
            has_purchased=('event_type', lambda x: int('purchase' in x.values)),
            distinct_categories_viewed=('category_1', lambda x: x[df['event_type'] == 'view'].nunique()),
            distinct_categories_added_to_cart=('category_1', lambda x: x[df['event_type'] == 'cart'].nunique()),
            distinct_brands_viewed=('brand', lambda x: x[df['event_type'] == 'view'].nunique()),
            distinct_brands_added_to_cart=('brand', lambda x: x[df['event_type'] == 'cart'].nunique()),
            distinct_products_viewed=('product_id', lambda x: x[df['event_type'] == 'view'].nunique()),
            distinct_products_added_to_cart=('product_id', lambda x: x[df['event_type'] == 'cart'].nunique())
        ).reset_index()

        # Crear el DataFrame `user_session_durations`
        user_session_durations = user_sessions[user_sessions['num_events'] > 1].copy()
        user_session_durations['session_duration'] = user_session_durations['session_end_time'] - user_session_durations['session_start_time']
        user_session_durations['avg_time_between_events'] = user_session_durations['session_duration'] / (user_session_durations['num_events'] - 1)

        # Crear el DataFrame `user_session_stats`
        user_session_stats = user_session_durations.groupby(['user_id', 'category_1']).agg(
            num_sessions=('user_session', 'nunique'),
            avg_events_per_session=('num_events', 'mean'),
            avg_session_duration=('session_duration', 'mean'),
            avg_time_between_events=('avg_time_between_events', 'mean')
        ).reset_index()

        # Crear el DataFrame `user_inter_session_times`
        user_inter_session_times = user_sessions.copy()
        user_inter_session_times['next_session_start_time'] = user_inter_session_times.groupby(['user_id', 'category_1'])['session_start_time'].shift(-1)

        # Crear el DataFrame `user_inter_session_durations`
        user_inter_session_durations = user_inter_session_times.dropna(subset=['next_session_start_time']).groupby(['user_id', 'category_1']).agg(
            avg_inter_session_time=('next_session_start_time', lambda x: (x - user_inter_session_times['session_start_time']).mean())
        ).reset_index()

        # Crear el DataFrame `repeated_views`
        repeated_views = df[df['event_type'] == 'view'].groupby(['user_id', 'category_1', 'product_id']).size().reset_index(name='view_count')
        repeated_views = repeated_views[repeated_views['view_count'] > 1]

        # Crear el DataFrame `user_repeated_views`
        user_repeated_views = repeated_views.groupby(['user_id', 'category_1']).agg(
            total_repeated_views=('view_count', 'sum')
        ).reset_index()

        # Crear el DataFrame final `users_behaviour_category`
        users_behaviour_category = user_sessions.groupby(['user_id', 'category_1']).agg(
            is_buyer=('has_purchased', 'max'),
            avg_distinct_categories_viewed=('distinct_categories_viewed', 'mean'),
            avg_distinct_categories_added_to_cart=('distinct_categories_added_to_cart', 'mean'),
            avg_distinct_brands_viewed=('distinct_brands_viewed', 'mean'),
            avg_distinct_brands_added_to_cart=('distinct_brands_added_to_cart', 'mean'),
            avg_distinct_products_viewed=('distinct_products_viewed', 'mean'),
            avg_distinct_products_added_to_cart=('distinct_products_added_to_cart', 'mean')
        ).reset_index()

        users_behaviour_category = users_behaviour_category.merge(user_session_stats, on=['user_id', 'category_1'], how='left')
        users_behaviour_category = users_behaviour_category.merge(user_inter_session_durations, on=['user_id', 'category_1'], how='left')
        users_behaviour_category = users_behaviour_category.merge(user_repeated_views, on=['user_id', 'category_1'], how='left')
        users_behaviour_category['total_repeated_views'] = users_behaviour_category['total_repeated_views'].fillna(0)


        # Aplicamos One-Hot Encoding a 'category_1'
        onehot_encoded = self.onehot_encoder.transform(users_behaviour_category[['category_1']])
        onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=self.onehot_encoder.get_feature_names_out(['category_1']), index=users_behaviour_category.index)
        
        # Unimos los datos codificados one-hot con el resto de los datos
        final_data = pd.concat([users_behaviour_category, onehot_encoded_df], axis=1)
        
        # Ordenamos las columnas para que coincidan con el orden esperado por el modelo
        expected_columns = ['num_sessions', 'avg_events_per_session', 'avg_time_between_events', 'avg_session_duration', 
                            'avg_inter_session_time', 'avg_distinct_categories_viewed', 'avg_distinct_categories_added_to_cart', 
                            'avg_distinct_brands_viewed', 'avg_distinct_brands_added_to_cart', 'avg_distinct_products_viewed', 
                            'avg_distinct_products_added_to_cart', 'total_repeated_views']
        expected_columns.extend(onehot_encoded_df.columns)

        # Eliminamos columnas no necesarias y ordenamos
        final_data = final_data[expected_columns]

        print(final_data)

        return final_data

    @classmethod
    def load_model(cls, filename):
        loaded_data = joblib.load(filename)
        instance = cls.__new__(cls)
        instance.model = loaded_data['model']
        instance.onehot_encoder = loaded_data['onehot_encoder']
        return instance
