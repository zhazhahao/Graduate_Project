document.getElementById('upload-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData();
    const fileInput = document.getElementById('query-image');
    const statusDiv = document.getElementById('status');
    
    if (!fileInput.files.length) {
        statusDiv.textContent = '请选择一张图片';
        statusDiv.style.backgroundColor = '#ffebee';
        return;
    }
    
    formData.append('query_image', fileInput.files[0]);
    statusDiv.textContent = '正在上传图片...';
    statusDiv.style.backgroundColor = '#e3f2fd';
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            statusDiv.textContent = data.error;
            statusDiv.style.backgroundColor = '#ffebee';
        } else {
            statusDiv.textContent = data.message;
            statusDiv.style.backgroundColor = '#e8f5e9';
            
            // 清空文件选择
            fileInput.value = '';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        statusDiv.textContent = '上传过程中发生错误';
        statusDiv.style.backgroundColor = '#ffebee';
    });
}); 