from django.urls import path
from . import views

urlpatterns = [
    # 页面路由
    path('', views.index, name='index'),
    path('generate/', views.generate, name='generate'),
    path('review/', views.review_view, name='review'),
    path('knowledge/', views.knowledge_view, name='knowledge'),
    path('case-review-detail/', views.case_review_detail, name='case_review_detail'),
    
    #知识库文件上传页面
    path('upload/', views.upload_single_file, name='upload_single_file'),

    
    # API路由
    path('api/test-case/<int:test_case_id>/', views.get_test_case, name='get_test_case'),
    path('api/update-test-case/', views.update_test_case, name='update_test_case'),#更新单个测试用例的状态到mysql
    path('api/generate/', views.generate_api, name='generate_api'),
    path('core/save-test-case/', views.save_test_case, name='save_test_case'),#批量保存大模型生成的测试用例
    path('api/review/', views.review_api, name='review_api'),#调用大模型对单个测试用例进行AI评审
    path('api/add-knowledge/', views.add_knowledge, name='add_knowledge'),
    path('api/knowledge-list/', views.knowledge_list, name='knowledge_list'),
    path('api/search-knowledge/', views.search_knowledge, name='search_knowledge'),   
] 