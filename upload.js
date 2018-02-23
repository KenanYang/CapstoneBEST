function uploadEx(pageN) {
    var canvas = document.getElementById("videoCanvas");
    var dataURL = canvas.toDataURL("image/png");
    document.getElementById('hidden_data').value = dataURL;
    var fd = new FormData(document.forms["form1"]);

    var xhr = new XMLHttpRequest();
    xhr.open('POST', 'upload/'+ pageN, true);

    xhr.upload.onprogress = function(e) {
        if (e.lengthComputable) {
            var percentComplete = (e.loaded / e.total) * 100;
            console.log(percentComplete + '% uploaded');
            /*
            alert('Successfully uploaded');
            */
        }
    };

    xhr.onload = function() {

    };
    xhr.send(fd);
};