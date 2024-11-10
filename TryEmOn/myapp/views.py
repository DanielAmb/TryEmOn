from django.shortcuts import render


def homepage(request):
    return render(request, 'new_index.html', {'title': 'TryEmOn'})
