// 主要JavaScript功能

document.addEventListener('DOMContentLoaded', function() {
    // 獲取元素
    const dreamInput = document.getElementById('dream-input');
    const dreamForm = document.getElementById('dream-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const charCount = document.getElementById('char-count');
    const processStatus = document.getElementById('process-status');
    const processDetail = document.getElementById('process-detail');
    const progressBar = document.getElementById('progress-bar');
    const errorMessage = document.getElementById('error-message');
    const restartBtn = document.getElementById('restart-btn');
    const shareBtn = document.getElementById('share-btn');
    
    if (!dreamInput || !dreamForm) return;
    
    // 處理進度的步驟（簡化版本）
    const steps = [
        { status: '正在分析夢境元素...', detail: '識別關鍵元素與象徵意義', progress: 10 },
        { status: '正在創作夢境故事...', detail: '融合夢境元素創作完整故事', progress: 30 },
        { status: '正在生成視覺圖像...', detail: '使用 Stable Diffusion 創建夢境視覺化圖像', progress: 70 },
        { status: '正在進行心理分析...', detail: '根據夢境內容進行深度分析', progress: 95 },
        { status: '完成！', detail: '您的夢境分析結果已經準備好', progress: 100 }
    ];
    
    // 字數計算
    dreamInput.addEventListener('input', function() {
        const count = dreamInput.value.length;
        charCount.textContent = count + ' 個字';
    });
    
    // 表單提交
    dreamForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const dreamText = dreamInput.value.trim();
        
        if (dreamText.length < 10) {
            errorMessage.textContent = '請輸入至少10個字的夢境描述';
            errorMessage.style.display = 'block';
            return;
        }
        
        // 隱藏錯誤訊息
        errorMessage.style.display = 'none';
        
        // 禁用提交按鈕
        analyzeBtn.disabled = true;
        
        // 顯示載入中
        loading.style.display = 'block';
        if (results) results.style.display = 'none';
        
        // 重置進度條
        progressBar.style.width = '0%';
        progressBar.setAttribute('aria-valuenow', 0);
        if (processDetail) processDetail.textContent = steps[0].detail;
        
        // 處理夢境
        processDream(dreamText);
    });
    
    // 如果有重新開始按鈕
    if (restartBtn) {
        restartBtn.addEventListener('click', function() {
            if (results) results.style.display = 'none';
            dreamInput.value = '';
            charCount.textContent = '0 個字';
            analyzeBtn.disabled = false;
            dreamInput.focus();
        });
    }
    
    // 如果有分享按鈕
    if (shareBtn) {
        shareBtn.addEventListener('click', function() {
            // 創建一個唯一的URL或是短連結
            const shareUrl = window.location.origin + '/share/' + Date.now();
            
            // 如果有模態框，設置連結並顯示模態框
            const shareLinkInput = document.getElementById('share-link');
            if (shareLinkInput) {
                shareLinkInput.value = shareUrl;
                
                // 如果使用Bootstrap的模態框
                const shareModal = new bootstrap.Modal(document.getElementById('shareModal'));
                if (shareModal) {
                    shareModal.show();
                }
            }
        });
        
        // 複製連結按鈕
        const copyLinkBtn = document.getElementById('copy-link-btn');
        if (copyLinkBtn) {
            copyLinkBtn.addEventListener('click', function() {
                const shareLink = document.getElementById('share-link');
                shareLink.select();
                document.execCommand('copy');
                
                // 顯示複製成功
                copyLinkBtn.textContent = '已複製!';
                setTimeout(function() {
                    copyLinkBtn.textContent = '複製';
                }, 2000);
            });
        }
        
        // 社交媒體分享按鈕
        const shareFacebookBtn = document.getElementById('share-facebook-btn');
        if (shareFacebookBtn) {
            shareFacebookBtn.addEventListener('click', function() {
                const shareUrl = document.getElementById('share-link').value;
                window.open('https://www.facebook.com/sharer/sharer.php?u=' + encodeURIComponent(shareUrl), '_blank');
            });
        }
        
        const shareTwitterBtn = document.getElementById('share-twitter-btn');
        if (shareTwitterBtn) {
            shareTwitterBtn.addEventListener('click', function() {
                const shareUrl = document.getElementById('share-link').value;
                const shareText = '我剛剛使用夢境分析系統分析了我的夢境，看看結果！';
                window.open('https://twitter.com/intent/tweet?text=' + encodeURIComponent(shareText) + '&url=' + encodeURIComponent(shareUrl), '_blank');
            });
        }
    }
    
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
            processStatus.textContent = step.status;
            if (processDetail) processDetail.textContent = step.detail;
            progressBar.style.width = step.progress + '%';
            progressBar.setAttribute('aria-valuenow', step.progress);
            
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
        fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ dream: dreamText }),
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || '處理請求時發生錯誤');
                });
            }
            return response.json();
        })
        .then(data => {
            // 確保進度條走完
            setTimeout(function() {
                clearInterval(progressInterval);
                processStatus.textContent = '完成！';
                if (processDetail) processDetail.textContent = '您的夢境分析結果已經準備好';
                progressBar.style.width = '100%';
                progressBar.setAttribute('aria-valuenow', 100);
                
                // 顯示結果
                displayResults(data);
                
                // 隱藏載入中
                setTimeout(function() {
                    loading.style.display = 'none';
                    analyzeBtn.disabled = false;
                    if (results) results.style.display = 'block';
                }, 500);
            }, Math.max(0, steps.length * 1200 - 1200));
        })
        .catch(error => {
            clearInterval(progressInterval);
            loading.style.display = 'none';
            analyzeBtn.disabled = false;
            
            errorMessage.textContent = error.message || '處理請求時發生錯誤';
            errorMessage.style.display = 'block';
        });
    }
    
    // 顯示結果（簡化版本）
    function displayResults(data) {
        if (!results) return;
        
        // 只填充完整故事、圖像和心理分析
        document.getElementById('final-story').textContent = data.finalStory;
        document.getElementById('psychology-analysis').textContent = data.psychologyAnalysis;
        
        // 設置圖像
        const dreamImage = document.getElementById('dream-image');
        if (data.imagePath) {
            dreamImage.src = data.imagePath;
            dreamImage.alt = '夢境視覺化圖像';
        } else {
            dreamImage.src = '/static/images/default_dream.png';
            dreamImage.alt = '未能生成夢境圖像';
        }
    }
});