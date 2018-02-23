function updateResult(pageN) {
    $.getJSON("updateResult/"+pageN,
            function (data) {
                document.getElementById('findR').innerHTML=data.Result;
            }
    );
}