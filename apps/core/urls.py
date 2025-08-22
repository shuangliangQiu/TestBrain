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
    #PRD分析页面
    path('analyser/', views.prd_analyser, name='prd_analyser'),
    #接口case生成页面
    path('api_case_generate/', views.api_case_generate, name='api_case_generate'),
    #文件下载
    path('download_file/', views.download_file, name='download_file'),
    
    # API路由
    path('api/copy-test-cases/', views.copy_test_cases, name='copy_test_cases'), #复制选中的用例集合
    path('api/export-test-cases-excel/', views.export_test_cases_excel, name='export_test_cases_excel'), #将用例集合导出到excel


    path('api/test-case/<int:test_case_id>/', views.get_test_case, name='get_test_case'),
    path('api/test-cases/<str:test_case_ids>/', views.get_test_cases, name='get_test_cases'),
    path('api/update-test-case/', views.update_test_case, name='update_test_case'),#更新单个测试用例的状态到mysql
    path('core/save-test-case/', views.save_test_case, name='save_test_case'),#批量保存大模型生成的测试用例
    path('api/review/', views.case_review, name='case_review'),#调用大模型对单个测试用例进行AI评审
    path('api/add-knowledge/', views.add_knowledge, name='add_knowledge'),
    path('api/knowledge-list/', views.knowledge_list, name='knowledge_list'),
    path('api/search-knowledge/', views.search_knowledge, name='search_knowledge'),   
    path('api/delete-test-cases/', views.delete_test_cases, name='delete_test_cases'), #删除选中的测试用例
] 