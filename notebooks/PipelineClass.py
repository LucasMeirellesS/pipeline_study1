# Pré-processamento para incluir no pipeline
from sklearn.impute import SimpleImputer

# Pipelines para tratar dados futuros
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class PipelineClass:
    ''' Classe para a estruturação de pipeline'''

    def __init__(self, num_features, cat_features, model, encoder, scaler):
        '''
            num_features: Colunas numericas do dataset a ser utilizado
            cat_features: colunas categoricas do dataset a ser utilizado
            model: O modelo utilizado no pipeline
            encoder: O decodificador de categoricas
            scaler: O normalizador de numericas
        '''
        self.num_features = num_features
        self.cat_features = cat_features
        self.model = model
        self.encoder = encoder
        self.scaler = scaler

    def set_encoder(self, encoder):
        ''' Setter do encoder'''
        self.encoder = encoder
    
    def set_model(self, model):
        '''Setter do Modelo'''
        self.model = model

    def set_scaler(self, scaler):
        '''Setter do Scaler'''
        self.scaler = scaler

    def imputer(self):
        '''Criei esse método para caso deseje fazer uma imputação em valores faltantes'''
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', self.scaler)
        ])
        self.scaler = numerical_transformer
        
    def get_preprocessor(self):
        '''Cria o preprossessor para codificar e normalizar as colunas categoricas e numéricas'''
        preprocessor = ColumnTransformer([
            ('encoder', self.encoder, self.cat_features),
            ('scaler', self.scaler, self.num_features)
        ])
        return preprocessor
    
    def get_pipeline(self):
        '''Gera o PipeLine para o modelo desejado'''
        model_pipeline = Pipeline(steps=[
            ('preprocessor', self.get_preprocessor()),
            ('model', self.model)
        ])

        return model_pipeline