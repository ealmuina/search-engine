$(function () {
    $('#content').find('> form').submit(function (event) {
        event.preventDefault();

        var query = $('#query-box')[0].value;
        var count = $('#count')[0].value;
        var beta = $('#beta-value')[0].value;
        var checked = $('input[type="checkbox"]:checked');

        var relevant = [];
        for (var i = 0; i < checked.length; i++) {
            relevant.push($(checked[i]).data('filename'));
        }

        $('#evaluation-results').load('/get_evaluations/', {
            'query': query,
            'count': count,
            'beta': beta,
            'relevant': relevant
        });
    });
});