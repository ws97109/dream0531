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
    const progressBar = document.getElementById('progress-bar');
    const errorMessage = document.getElementById('error-message');
    const restartBtn = document.getElementById('restart-btn');
    
    if (!dreamInput || !dreamForm) return;
    
    // 處理進度的步驟
    const steps = [
        { status: '正在分析夢境元素...', progress: 10 },
        { status: '正在生成故事架構...', progress: 20 },
        { status: '正在創作初步故事...', progress: 35 },
        { status: '正在評估故事品質...', progress: 50 },
        { status: '正在優化故事內容...', progress: 65 },
        { status: '正在翻譯故事...', progress: 75 },
        { status: '正在生成視覺圖像...', progress: 85 },
        { status: '正在進行心理分析...', progress: 95 },
        { status: '完成！', progress: 100 }
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
    
    // 處理夢境分析
    function processDream(dreamText) {
        // 模擬進度更新
        let currentStep = 0;
        
        const progressInterval = setInterval(function() {
            if (currentStep >= steps.length) {
                clearInterval(progressInterval);
                return;
            }
            
            const step = steps[currentStep];
            processStatus.textContent = step.status;
            progressBar.style.width = step.progress + '%';
            progressBar.setAttribute('aria-valuenow', step.progress);
            
            currentStep++;
        }, 1000);
        
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
            }, Math.max(0, steps.length * 1000 - 2000));
        })
        .catch(error => {
            clearInterval(progressInterval);
            loading.style.display = 'none';
            analyzeBtn.disabled = false;
            
            errorMessage.textContent = error.message || '處理請求時發生錯誤';
            errorMessage.style.display = 'block';
        });
    }
    
    // 顯示結果
    function displayResults(data) {
        if (!results) return;
        
        // 填充結果
        document.getElementById('initial-story').textContent = data.initialStory;
        document.getElementById('story-feedback').textContent = data.storyFeedback;
        document.getElementById('final-story').textContent = data.finalStory;
        document.getElementById('translation').textContent = data.translation;
        document.getElementById('psychology-analysis').textContent = data.psychologyAnalysis;
        
        // 設置圖像
        const dreamImage = document.getElementById('dream-image');
        if (data.imagePath) {
            dreamImage.src = data.imagePath;
            dreamImage.alt = '夢境視覺化';
        } else {
            dreamImage.src = '/static/images/default_dream.png';
            dreamImage.alt = '未生成夢境圖像';
        }
    }
});