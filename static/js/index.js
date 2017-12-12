$(function () {
    $.get('/suggest/', {}, function (data) {
        $('#content').html(data);
    });
});