{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-lg-8 offset-lg-2">
        <div class="card">
            <div class="card-header bg-primary text-white">
                <h2 class="card-title mb-0">夢境分析與視覺化</h2>
            </div>
            <div class="card-body">
                <div class="input-section">
                    <form id="dream-form">
                        <div class="mb-3">
                            <label for="dream-input" class="form-label">請輸入您的夢境片段：</label>
                            <textarea id="dream-input" name="dream" class="form-control" 
                                rows="5" placeholder="例如：我夢見自己在飛行，下方是一片湛藍的大海，突然看到一座閃閃發光的城堡..." required></textarea>
                            <div class="form-text" id="char-count">0 個字</div>
                        </div>
                        <div class="text-center">
                            <button type="submit" class="btn btn-primary" id="analyze-btn">開始分析</button>
                        </div>
                    </form>
                </div>
                
                <div class="loading mt-4" id="loading" style="display: none;">
                    <div class="text-center mb-3">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">處理中...</span>
                        </div>
                    </div>
                    <p class="text-center" id="process-status">正在分析夢境元素...</p>
                    <div class="progress">
                        <div class="progress-bar progress-bar-striped progress-bar-animated" id="progress-bar" role="progressbar" style="width: 0%;" 
                            aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
                    </div>
                    <p class="text-center text-muted mt-2 small" id="process-detail">分析夢境元素並創建故事架構...</p>
                </div>
                
                <div class="alert alert-danger mt-3" id="error-message" style="display: none;"></div>
            </div>
        </div>
        
        <div class="results-section mt-4" id="results" style="display: none;">
            <div class="card mb-4">
                <div class="card-header bg-success text-white">
                    <h5 class="mb-0">夢境故事</h5>
                </div>
                <div class="card-body">
                    <p id="final-story"></p>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="card mb-4">
                        <div class="card-header bg-primary text-white">
                            <h5 class="mb-0">視覺化夢境</h5>
                        </div>
                        <div class="card-body text-center">
                            <img id="dream-image" class="img-fluid rounded" src="" alt="視覺化夢境圖像">
                            <div class="text-muted small mt-2">由 Stable Diffusion 生成</div>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="card mb-4">
                        <div class="card-header bg-danger text-white">
                            <h5 class="mb-0">夢境心理分析</h5>
                        </div>
                        <div class="card-body">
                            <p id="psychology-analysis"></p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="text-center mb-4">
                <button id="restart-btn" class="btn btn-primary">重新開始</button>
                <button id="share-btn" class="btn btn-success ms-2">分享結果</button>
            </div>
        </div>
    </div>
</div>

<!-- 分享模態框 -->
<div class="modal fade" id="shareModal" tabindex="-1" aria-labelledby="shareModalLabel" aria-hidden="true">
    <div class="modal-dialog">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title" id="shareModalLabel">分享您的夢境分析結果</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body">
                <div class="mb-3">
                    <label for="share-link" class="form-label">分享連結</label>
                    <div class="input-group">
                        <input type="text" class="form-control" id="share-link" readonly>
                        <button class="btn btn-outline-secondary" type="button" id="copy-link-btn">複製</button>
                    </div>
                </div>
                <div class="d-grid gap-2 mt-3">
                    <button type="button" class="btn btn-primary" id="share-facebook-btn">
                        <i class="bi bi-facebook"></i> 分享到 Facebook
                    </button>
                    <button type="button" class="btn btn-info text-white" id="share-twitter-btn">
                        <i class="bi bi-twitter"></i> 分享到 Twitter
                    </button>
                </div>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">關閉</button>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
    $(document).ready(function() {
        const dreamInput = $('#dream-input');
        const dreamForm = $('#dream-form');
        const analyzeBtn = $('#analyze-btn');
        const loading = $('#loading');
        const results = $('#results');
        const charCount = $('#char-count');
        const processStatus = $('#process-status');
        const processDetail = $('#process-detail');
        const progressBar = $('#progress-bar');
        const errorMessage = $('#error-message');
        const restartBtn = $('#restart-btn');
        const shareBtn = $('#share-btn');
        const copyLinkBtn = $('#copy-link-btn');
        const shareFacebookBtn = $('#share-facebook-btn');
        const shareTwitterBtn = $('#share-twitter-btn');
        
        // 防重複提交狀態
        let isProcessing = false;
        
        // 處理進度的步驟
        const steps = [
            { status: '正在分析夢境元素...', detail: '識別關鍵元素與象徵意義', progress: 10 },
            { status: '正在創作夢境故事...', detail: '融合夢境元素創作完整故事', progress: 30 },
            { status: '正在生成視覺圖像...', detail: '使用 Stable Diffusion 創建夢境視覺化圖像', progress: 70 },
            { status: '正在進行心理分析...', detail: '根據夢境內容進行深度分析', progress: 95 },
            { status: '完成！', detail: '您的夢境分析結果已經準備好', progress: 100 }
        ];
        
        // 字數計算
        dreamInput.on('input', function() {
            const count = dreamInput.val().length;
            charCount.text(count + ' 個字');
        });
        
        // 表單提交
        dreamForm.on('submit', function(e) {
            e.preventDefault();
            
            const dreamText = dreamInput.val().trim();
            
            // 基本驗證
            if (dreamText.length < 10) {
                errorMessage.text('請輸入至少10個字的夢境描述');
                errorMessage.show();
                return;
            }
            
            // 簡單的重複提交檢查
            if (analyzeBtn.prop('disabled')) {
                return;
            }
            
            // 隱藏錯誤訊息
            errorMessage.hide();
            
            // 禁用提交按鈕並改變文字
            analyzeBtn.prop('disabled', true);
            analyzeBtn.text('分析中...');
            
            // 顯示載入中
            loading.show();
            results.hide();
            
            // 重置進度條
            progressBar.css('width', '0%');
            progressBar.attr('aria-valuenow', 0);
            processDetail.text(steps[0].detail);
            
            // 使用AJAX發送請求
            processDream(dreamText);
        });
        
        // 重新開始按鈕
        restartBtn.on('click', function() {
            // 重置狀態
            analyzeBtn.prop('disabled', false);
            analyzeBtn.text('開始分析');
            
            results.hide();
            dreamInput.val('');
            charCount.text('0 個字');
            errorMessage.hide();
            dreamInput.focus();
        });
        
        // 分享按鈕
        shareBtn.on('click', function() {
            // 創建一個唯一的URL或是短連結
            const shareUrl = window.location.origin + '/share/' + Date.now();
            $('#share-link').val(shareUrl);
            
            // 顯示分享模態框
            var shareModal = new bootstrap.Modal(document.getElementById('shareModal'));
            shareModal.show();
        });
        
        // 複製連結
        copyLinkBtn.on('click', function() {
            const shareLink = $('#share-link');
            shareLink.select();
            document.execCommand('copy');
            
            // 顯示複製成功
            copyLinkBtn.text('已複製!');
            setTimeout(function() {
                copyLinkBtn.text('複製');
            }, 2000);
        });
        
        // 社交媒體分享按鈕
        shareFacebookBtn.on('click', function() {
            const shareUrl = $('#share-link').val();
            window.open('https://www.facebook.com/sharer/sharer.php?u=' + encodeURIComponent(shareUrl), '_blank');
        });
        
        shareTwitterBtn.on('click', function() {
            const shareUrl = $('#share-link').val();
            const shareText = '我剛剛使用夢境分析系統分析了我的夢境，看看結果！';
            window.open('https://twitter.com/intent/tweet?text=' + encodeURIComponent(shareText) + '&url=' + encodeURIComponent(shareUrl), '_blank');
        });
        
        // 處理夢境分析
        function processDream(dreamText) {
            // 進度更新
            let currentStep = 0;
            
            const progressInterval = setInterval(function() {
                if (currentStep >= steps.length) {
                    clearInterval(progressInterval);
                    return;
                }
                
                const step = steps[currentStep];
                processStatus.text(step.status);
                processDetail.text(step.detail);
                progressBar.css('width', step.progress + '%');
                progressBar.attr('aria-valuenow', step.progress);
                
                // 圖像生成步驟需要停留更長時間
                if (step.status.includes('生成視覺圖像')) {
                    setTimeout(function() {
                        currentStep++;
                    }, 2000); // 多等待2秒
                } else {
                    currentStep++;
                }
            }, 1200); // 稍微調慢進度條速度
            
            // 發送API請求
            $.ajax({
                url: '/api/analyze',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ dream: dreamText }),
                timeout: 180000,  // 設定3分鐘超時
                success: function(response) {
                    // 確保進度條走完
                    setTimeout(function() {
                        clearInterval(progressInterval);
                        processStatus.text('完成！');
                        processDetail.text('您的夢境分析結果已經準備好');
                        progressBar.css('width', '100%');
                        progressBar.attr('aria-valuenow', 100);
                        
                        // 顯示結果
                        displayResults(response);
                        
                        // 隱藏載入中並恢復狀態
                        setTimeout(function() {
                            loading.hide();
                            results.show();
                            // 恢復按鈕狀態
                            analyzeBtn.prop('disabled', false);
                            analyzeBtn.text('開始分析');
                        }, 500);
                    }, Math.max(0, steps.length * 1200 - 1200));
                },
                error: function(xhr, status, error) {
                    clearInterval(progressInterval);
                    loading.hide();
                    
                    // 恢復按鈕狀態
                    analyzeBtn.prop('disabled', false);
                    analyzeBtn.text('開始分析');
                    
                    let errorMsg = '處理請求時發生錯誤';
                    if (xhr.responseJSON && xhr.responseJSON.error) {
                        errorMsg = xhr.responseJSON.error;
                    } else if (status === 'timeout') {
                        errorMsg = '請求超時，請重試';
                    }
                    
                    errorMessage.text(errorMsg);
                    errorMessage.show();
                }
            });
        }
        
        // 顯示結果
        function displayResults(data) {
            // 填充完整故事、圖像和心理分析
            $('#final-story').text(data.finalStory);
            $('#psychology-analysis').text(data.psychologyAnalysis);
            
            // 設置圖像
            if (data.imagePath) {
                $('#dream-image').attr('src', data.imagePath);
                $('#dream-image').attr('alt', '夢境視覺化圖像');
            } else {
                $('#dream-image').attr('src', '/static/images/default_dream.png');
                $('#dream-image').attr('alt', '未能生成夢境圖像');
            }
        }
    });
</script>
{% endblock %}