from warrant import Cognito
from data.const import user_pool_id, client_id, region

user = Cognito(user_pool_id, client_id, user_pool_region=region)


def login(user_name, password):
    try:
        user.authenticate(user_name, password)
        return True, user
    except Exception as e:
        st.error(f"Login failed : {str(e)}")
        return False, None
